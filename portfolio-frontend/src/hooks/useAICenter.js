import { useQuery } from '@tanstack/react-query';
import api from '../api/api';

const fetchAICenter = async () => {
  const res = await api.get('ai-center/');
  return res.data || {};
};

export const useAICenter = () => useQuery({
  queryKey: ['ai-center-data'],
  queryFn: fetchAICenter,
  staleTime: 60_000,
  refetchInterval: 30_000,
});
