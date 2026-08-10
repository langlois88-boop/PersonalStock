"""
Tests de régression pour les 3 bugs trouvés lors du premier scan réel en
prod le 2026-08-09 (clés Anthropic/Ollama enfin valides, révélant des
bugs invisibles jusque-là faute de pouvoir réellement appeler les API) :

1. deepseek_triage.py appelait en dur OLLAMA_BASE_URL + "/api/generate"
   (endpoint natif Ollama, absent sur l'instance LocalAI du NAS -> 404),
   sans suivre la même bascule chat_mode que ai_advisor.py::DeepSeekAdvisor.
2. claude_verifier.py faisait un json.loads() naïf sur la réponse Claude,
   qui l'enveloppe fréquemment dans un bloc markdown ```json ... ```.

(Le 3e bug trouvé ce soir-là, les clés Alpaca invalides, est une panne
externe de credentials -- rien à tester ici, aucune ligne de code n'était
en cause.)
"""

from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, TestCase

from .services import deepseek_triage, claude_verifier
from .services.llm_json import extract_json_object


class ExtractJsonObjectTests(SimpleTestCase):
    """Le coeur du fix : extract_json_object doit survivre à ce que les
    LLM font réellement, pas seulement à ce qu'on leur demande de faire."""

    def test_raw_json_still_works(self):
        self.assertEqual(
            extract_json_object('{"verdict": "confirmed", "reasoning": "ok"}'),
            {"verdict": "confirmed", "reasoning": "ok"},
        )

    def test_markdown_fenced_json_with_language_tag(self):
        # Exactement la forme observée en prod le 2026-08-09 sur BRCC.
        raw = '```json\n{\n  "verdict": "uncertain",\n  "reasoning": "test"\n}\n```'
        self.assertEqual(
            extract_json_object(raw),
            {"verdict": "uncertain", "reasoning": "test"},
        )

    def test_markdown_fenced_json_without_language_tag(self):
        raw = '```\n{"verdict": "rejected", "reasoning": "test"}\n```'
        self.assertEqual(extract_json_object(raw)["verdict"], "rejected")

    def test_think_block_from_reasoning_models(self):
        # DeepSeek-R1 (voir ai_advisor.py::stream_answer, qui gère déjà ça
        # pour le chat interactif) préfixe parfois sa réponse d'un bloc de
        # raisonnement explicite.
        raw = '<think>Le ticker semble correct, je vais dire no_reason.</think>\n{"verdict": "no_reason", "reasoning": "rien trouvé"}'
        self.assertEqual(extract_json_object(raw)["verdict"], "no_reason")

    def test_think_block_and_fence_combined(self):
        raw = '<think>...</think>\n```json\n{"verdict": "uncertain", "reasoning": "x"}\n```'
        self.assertEqual(extract_json_object(raw)["verdict"], "uncertain")

    def test_genuinely_invalid_json_still_raises(self):
        with self.assertRaises(ValueError):
            extract_json_object("ceci n'est pas du json")

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            extract_json_object("")


class DeepSeekTriageEndpointRoutingTests(TestCase):
    """Bug #1 : confirme que triage_ticker() appelle le bon endpoint selon
    OLLAMA_CHAT_MODE, exactement comme ai_advisor.py::DeepSeekAdvisor."""

    def setUp(self):
        patcher = patch.object(deepseek_triage, "fetch_recent_news", return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _fake_sse_response(full_text: str):
        """Simule une réponse streaming SSE comme celle réellement servie
        par LocalAI (confirmé par curl direct le 2026-08-09 : stream=False
        ne répond jamais sur cette instance, seul stream=True fonctionne).
        Découpe le texte en 2 chunks pour vérifier que la concaténation
        fonctionne, pas juste un chunk unique par accident."""
        import json as _json
        mid = max(1, len(full_text) // 2)
        lines = [
            f"data: {_json.dumps({'choices': [{'delta': {'content': full_text[:mid]}}]})}",
            f"data: {_json.dumps({'choices': [{'delta': {'content': full_text[mid:]}}]})}",
            "data: [DONE]",
        ]
        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None
        fake_response.iter_lines.return_value = iter(lines)
        return fake_response

    def test_chat_mode_enabled_uses_v1_chat_completions(self):
        """Reproduit la config réelle du NAS (OLLAMA_CHAT_MODE=1) -- avant
        le fix, ce test aurait appelé /api/generate, pas le
        /chat/completions moqué ici."""
        fake_response = self._fake_sse_response('{"verdict": "no_reason", "reasoning": "rien"}')
        with patch.object(deepseek_triage, "OLLAMA_CHAT_MODE", True), \
                patch.object(deepseek_triage, "OLLAMA_CHAT_BASE_URL", "http://fake-nas:8090/v1"), \
                patch.object(deepseek_triage, "OLLAMA_BASE_URL", "http://fake-nas:11434"), \
                patch.object(deepseek_triage.requests, "post", return_value=fake_response) as mock_post:
            result = deepseek_triage.triage_ticker("AAPL")

        called_url = mock_post.call_args[0][0]
        self.assertEqual(called_url, "http://fake-nas:8090/v1/chat/completions")
        self.assertIn("messages", mock_post.call_args[1]["json"])
        self.assertTrue(mock_post.call_args[1]["json"]["stream"])
        self.assertTrue(mock_post.call_args[1].get("stream"))
        self.assertEqual(result.verdict, "no_reason")

    def test_chat_mode_response_is_consumed_as_sse_stream_not_json(self):
        """Non-régression directe du bug de timeout trouvé le 2026-08-09
        après le premier fix : stream=False hangait sur cette instance
        LocalAI (HTTP 000 après 15s en curl direct). Confirme que le code
        lit bien resp.iter_lines(), pas resp.json() -- et que l'encodage
        est forcé en UTF-8 avant lecture (sinon les accents se corrompent
        en Ã©-style mojibake, confirmé en prod le 2026-08-09 sur GOOGL :
        LocalAI ne renvoie pas de charset explicite pour
        text/event-stream, requests devine ISO-8859-1 sans ce réglage —
        voir ai_advisor.py::stream_answer qui fait déjà ce même réglage)."""
        fake_response = self._fake_sse_response('{"verdict": "uncertain", "reasoning": "test stream"}')
        with patch.object(deepseek_triage, "OLLAMA_CHAT_MODE", True), \
                patch.object(deepseek_triage, "OLLAMA_CHAT_BASE_URL", "http://fake-nas:8090/v1"), \
                patch.object(deepseek_triage, "OLLAMA_BASE_URL", "http://fake-nas:11434"), \
                patch.object(deepseek_triage.requests, "post", return_value=fake_response):
            result = deepseek_triage.triage_ticker("AAPL")

        self.assertEqual(fake_response.encoding, "utf-8")

        fake_response.iter_lines.assert_called_once()
        fake_response.json.assert_not_called()
        self.assertEqual(result.reasoning, "test stream")

    def test_chat_mode_disabled_uses_native_api_generate(self):
        """Le chemin legacy (Ollama natif, sans LocalAI) doit continuer à
        fonctionner pour ne pas casser un déploiement qui n'utilise pas
        OLLAMA_CHAT_MODE."""
        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None
        fake_response.json.return_value = {"response": '{"verdict": "uncertain", "reasoning": "rien"}'}
        with patch.object(deepseek_triage, "OLLAMA_CHAT_MODE", False), \
                patch.object(deepseek_triage, "OLLAMA_CHAT_BASE_URL", "http://fake-nas:11434"), \
                patch.object(deepseek_triage, "OLLAMA_BASE_URL", "http://fake-nas:11434"), \
                patch.object(deepseek_triage.requests, "post", return_value=fake_response) as mock_post:
            result = deepseek_triage.triage_ticker("AAPL")

        called_url = mock_post.call_args[0][0]
        self.assertEqual(called_url, "http://fake-nas:11434/api/generate")
        self.assertEqual(result.verdict, "uncertain")

    def test_404_on_native_endpoint_falls_back_to_uncertain_not_crash(self):
        """Reproduit exactement le symptôme observé en prod : /api/generate
        répond 404 -> le pipeline ne doit pas planter, juste dégrader."""
        import requests as real_requests
        fake_response = MagicMock()
        fake_response.raise_for_status.side_effect = real_requests.exceptions.HTTPError("404 Client Error")
        with patch.object(deepseek_triage, "OLLAMA_CHAT_MODE", False), \
                patch.object(deepseek_triage, "OLLAMA_BASE_URL", "http://fake-nas:11434"), \
                patch.object(deepseek_triage, "OLLAMA_CHAT_BASE_URL", "http://fake-nas:11434"), \
                patch.object(deepseek_triage.requests, "post", return_value=fake_response):
            result = deepseek_triage.triage_ticker("AAPL")

        self.assertEqual(result.verdict, "uncertain")


class ClaudeVerifierToolUseTests(TestCase):
    """Historique des 3 approches essayées le 2026-08-09 pour éviter le
    JSON malformé (bug initial trouvé sur BRCC) :
    1. JSON en texte libre + extract_json_object() -- fonctionnait mais
       fragile (```json fences, ou max_tokens épuisé par le raisonnement
       étendu avant d'écrire le JSON).
    2. Préremplissage (message assistant "{") -- rejeté par l'API,
       "This model does not support assistant message prefill".
    3. Tool use forcé (tool_choice) -- solution retenue : Claude ne peut
       littéralement répondre que dans le schéma défini, réponse déjà
       parsée en dict par le SDK (.input), pas de conflit avec le
       raisonnement étendu. Confirmé en direct : 3/3 essais réussis sur
       NVDA (contre 3/4 avec l'approche JSON texte libre)."""

    def setUp(self):
        patcher = patch.object(claude_verifier, "fetch_recent_news", return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)

    def _fake_tool_use_response(self, verdict="confirmed", reasoning="ok", confidence="high",
                                 input_tokens=100, output_tokens=50):
        block = MagicMock()
        block.type = "tool_use"
        block.input = {"verdict": verdict, "reasoning": reasoning, "confidence": confidence}
        response = MagicMock()
        response.content = [block]
        response.stop_reason = "tool_use"
        response.usage.input_tokens = input_tokens
        response.usage.output_tokens = output_tokens
        return response

    def test_tool_choice_forces_submit_verdict(self):
        """Le coeur du fix : Claude doit être forcé à utiliser l'outil
        submit_verdict, pas juste invité à répondre en JSON."""
        fake_response = self._fake_tool_use_response(verdict="rejected", reasoning="raison trouvée")
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response
        with patch.object(claude_verifier, "ANTHROPIC_API_KEY", "sk-ant-fake"), \
                patch.object(claude_verifier.anthropic, "Anthropic", return_value=fake_client):
            result = claude_verifier.verify_ticker("AAPL", {"pe_ratio": 15})

        call_kwargs = fake_client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["tool_choice"], {"type": "tool", "name": "submit_verdict"})
        self.assertEqual(call_kwargs["tools"][0]["name"], "submit_verdict")
        self.assertEqual(result.verdict, "rejected")
        self.assertEqual(result.reasoning, "raison trouvée")

    def test_no_temperature_param_sent(self):
        """Non-régression : `temperature` est déprécié pour ce modèle
        ('`temperature` is deprecated for this model.') -- confirmé en
        prod le 2026-08-09 après une tentative de fix pour réduire la
        variance entre appels. Ne doit plus jamais être envoyé."""
        fake_response = self._fake_tool_use_response()
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response
        with patch.object(claude_verifier, "ANTHROPIC_API_KEY", "sk-ant-fake"), \
                patch.object(claude_verifier.anthropic, "Anthropic", return_value=fake_client):
            claude_verifier.verify_ticker("AAPL", {"pe_ratio": 15})

        self.assertNotIn("temperature", fake_client.messages.create.call_args[1])

    def test_no_assistant_prefill_message_sent(self):
        """Non-régression : claude-sonnet-5 rejette le préremplissage
        ('This model does not support assistant message prefill') --
        confirmé en prod le 2026-08-09. Un seul message, role=user."""
        fake_response = self._fake_tool_use_response()
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response
        with patch.object(claude_verifier, "ANTHROPIC_API_KEY", "sk-ant-fake"), \
                patch.object(claude_verifier.anthropic, "Anthropic", return_value=fake_client):
            claude_verifier.verify_ticker("AAPL", {"pe_ratio": 15})

        messages_sent = fake_client.messages.create.call_args[1]["messages"]
        self.assertEqual(len(messages_sent), 1)
        self.assertEqual(messages_sent[0]["role"], "user")

    def test_tool_input_is_used_directly_no_json_parsing_needed(self):
        """block.input est déjà un dict fourni par le SDK -- confirme
        qu'on ne fait plus de json.loads()/extract_json_object() dessus
        (donc plus aucun risque de JSON malformé, le bug d'origine)."""
        fake_response = self._fake_tool_use_response(
            verdict="uncertain", reasoning="signal mixte", confidence="low",
            input_tokens=1543, output_tokens=267,
        )
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response
        with patch.object(claude_verifier, "ANTHROPIC_API_KEY", "sk-ant-fake"), \
                patch.object(claude_verifier.anthropic, "Anthropic", return_value=fake_client):
            result = claude_verifier.verify_ticker("AAPL", {"pe_ratio": 15})

        self.assertEqual(result.verdict, "uncertain")
        self.assertEqual(result.reasoning, "signal mixte")
        self.assertEqual(result.confidence, "low")
        self.assertEqual(result.tokens_used, 1543 + 267)

    def test_missing_tool_use_block_degrades_to_uncertain_not_crash(self):
        """Défensif : même avec tool_choice forcé, si jamais aucun bloc
        tool_use n'apparaît (ex: réponse tronquée par max_tokens avant
        l'appel d'outil, cas analogue au bug NVDA du 2026-08-09 avec
        l'ancienne approche texte), le pipeline ne doit jamais planter."""
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        response = MagicMock()
        response.content = [thinking_block]  # aucun bloc "tool_use"
        response.stop_reason = "max_tokens"
        response.usage.input_tokens = 500
        response.usage.output_tokens = 2000
        fake_client = MagicMock()
        fake_client.messages.create.return_value = response
        with patch.object(claude_verifier, "ANTHROPIC_API_KEY", "sk-ant-fake"), \
                patch.object(claude_verifier.anthropic, "Anthropic", return_value=fake_client):
            result = claude_verifier.verify_ticker("NVDA", {"pe_ratio": 45})

        self.assertEqual(result.verdict, "uncertain")
        self.assertIn("Erreur technique", result.reasoning)

    def test_max_tokens_has_headroom_for_thinking_plus_answer(self):
        """Non-régression sur la valeur : 400 (originale) s'est révélé
        insuffisant en prod le 2026-08-09 (raisonnement étendu épuisant
        le budget avant la réponse). Verrouille qu'on ne redescend pas
        accidentellement en dessous."""
        fake_response = self._fake_tool_use_response()
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response
        with patch.object(claude_verifier, "ANTHROPIC_API_KEY", "sk-ant-fake"), \
                patch.object(claude_verifier.anthropic, "Anthropic", return_value=fake_client):
            claude_verifier.verify_ticker("AAPL", {"pe_ratio": 15})

        self.assertGreaterEqual(fake_client.messages.create.call_args[1]["max_tokens"], 2000)
