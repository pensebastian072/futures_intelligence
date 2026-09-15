# Futures Intelligence Agent Rules

- This repository is research-only and SHADOW. Never add broker execution, order routing, or account access.
- Never persist or print credentials. Secrets come from environment variables or gitignored local files.
- Metered data must be stored immutably before transformation and reused by request hash.
- Do not use a continuous futures series as a substitute for dated-contract curves.
- Every observation is joined by `available_at`; a calendar date alone is not point-in-time evidence.
- Missing, stale, or unavailable validation fails closed to `promoted=false`.
- Reuse `C:\Users\<your-user>\copper_brain\copper_brain\validate.py`; never copy its PBO/DSR math.
- Use `.venv\Scripts\python.exe`, keep PowerShell scripts ASCII-only, and use pip truststore.

