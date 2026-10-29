# SpamGuard

Machine-learning spam detection web app: paste an email or SMS and get a **SPAM / HAM verdict**, a **risk score**, the **words that drove the decision**, a **suspicious-content checklist**, a saved **history**, and a live **dashboard**.

**Stack:** Flask · scikit-learn (TF-IDF + Multinomial Naive Bayes) · SQLite · vanilla HTML/CSS/JS (no build step, no CDN scripts).

## Original project vs. what was added

| Original (preserved) | Added |
|---|---|
| Flask app, `POST /predict`, `/evaluation` | Landing page, scanner page, dashboard, history page, redesigned evaluation page |
| TF-IDF (3000) + MultinomialNB on `spam.csv`, same 80/20 stratified split | Model wrapped in `ml.py`; per-word explanation from the model's learned weights |
| Spam/ham sample buttons, dark-mode toggle, confusion matrix, classification report | SQLite database, signup/login/logout (hashed passwords), CSRF, rate limiting, security headers |
| `spam_detection.py` CLI walkthrough (untouched) | Search/filter/delete/export-CSV history; 14-day activity and score charts from real scans |

`legacy/app_original.py` is the original `app.py`, kept for reference. The original `/predict` URL still works.

> The original project had **no** login, database or dashboard. Accounts and persistence were added so the dashboard can show real data. Guests can use everything; their scans are tied to their browser and move to their account if they sign up.

## Run on Windows (VS Code)

Requires **Python 3.10+** (tick "Add Python to PATH" when installing).

1. Unzip, then in VS Code choose **File → Open Folder…** and select the `SpamGuard` folder.
2. Open the terminal (**Ctrl + `**), using **Command Prompt** or **PowerShell**.
3. One-time setup:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   copy .env.example .env
   ```
   If PowerShell blocks `activate`, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (or use Command Prompt).
4. Generate a secret key and paste it after `SECRET_KEY=` in `.env`:
   ```
   python -c "import secrets; print(secrets.token_hex(32))"
   ```
5. Start the app:
   ```
   python app.py
   ```
6. Open **http://127.0.0.1:5000**

`setup.bat` and `run.bat` do steps 3 and 5 for you. In VS Code, select the interpreter with **Ctrl+Shift+P → Python: Select Interpreter → .\venv\Scripts\python.exe**.

### Tests
```
python -m unittest discover -s tests -v
```
20 integration tests covering prediction, validation, CSRF, auth, per-user data isolation, CSV safety and rate limiting. No extra packages needed.

## Project structure
```
app.py            Flask app: pages, auth, JSON API, security
ml.py             Dataset load, training, evaluation metrics, prediction + explanation
analysis.py       Rule-based checks (links, phone numbers, prize wording, urgency, credentials, caps, !!!)
database.py       SQLite schema and connection (users, scans)
spam.csv          SMS Spam Collection dataset
spam_detection.py Original step-by-step CLI script
templates/        Jinja pages
static/           CSS design system, JS (scanner, dashboard SVG charts, history), images
tests/            Integration tests
legacy/           Original app.py
```

## API
POST/DELETE requests need the `X-CSRF-Token` header (value from the page's `<meta name="csrf-token">`).

| Endpoint | Purpose |
|---|---|
| `POST /api/predict` (also `/predict`) | `{subject?, message}` → verdict, score, risk, word evidence, checklist; saves to history |
| `GET /api/history?page&per_page&label&q` | Paginated, searchable history |
| `DELETE /api/history/<id>` · `DELETE /api/history` | Delete one / all of your scans |
| `GET /history/export.csv` | Download your history |
| `GET /api/stats` | Dashboard numbers (totals, 14-day timeline, score bands, top trigger words, recent scans) |
| `GET /api/health` | Health check |

## Security notes
Passwords hashed with scrypt (Werkzeug) · CSRF token on every state-changing request · HttpOnly, SameSite=Lax session cookies (Secure with HTTPS) · login and predict rate limiting · strict CSP and security headers · parameterised SQL · output inserted as text only (no XSS) · open-redirect-safe `next` · CSV formula-injection guard · input length limits · generic login errors.

Before deploying publicly: set a real `SECRET_KEY`, set `SESSION_COOKIE_SECURE=1`, serve over HTTPS, and use a production server instead of `python app.py`, for example `pip install waitress` then `waitress-serve --port=8000 app:app`. The in-memory rate limiter is per-process; use shared storage (e.g. Redis) for multi-worker setups.

## Limitations
- The model is trained on the **SMS** Spam Collection, so long modern phishing emails may be classified less reliably. Held-out results: accuracy ≈ 96.7 %, spam precision ≈ 99.0 %, spam recall ≈ 74.8 %. It rarely flags good messages but misses about 1 in 4 spam messages.
- The checklist is rule-based and shown beside the model verdict; it does not change the score.
- Fonts load from Google Fonts; offline they fall back to system fonts and everything else still works.
- Scan text is stored in `instance/spamguard.db` on the machine running the app.
