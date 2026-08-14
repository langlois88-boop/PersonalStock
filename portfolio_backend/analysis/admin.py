from django.contrib import admin

from .models import (
    ScreenerPreset,
    ScanRun,
    ScanResult,
    FundamentalLabPosition,
    RejectionLog,
    RejectionOutcome,
    LabWeeklySuggestion,
)


admin.site.register(ScreenerPreset)
admin.site.register(ScanRun)
admin.site.register(ScanResult)
admin.site.register(FundamentalLabPosition)
admin.site.register(RejectionLog)
admin.site.register(RejectionOutcome)
admin.site.register(LabWeeklySuggestion)
