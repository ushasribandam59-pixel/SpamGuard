"""SpamGuard - Flask web application.

Run:  python app.py        (development)
      flask --app app run  (alternative)
"""
from __future__ import annotations

import collections
import csv
import hmac
import io
import json
import logging
import os
import re
import secrets
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

from flask import (Flask, Response, abort, flash, g, jsonify, redirect,
                   render_template, request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

try:  # optional: load variables from a .env file if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:  # pragma: no cover
    pass

from analysis import analyze_signals
from database import close_db, get_db, init_db
from ml import SpamModel

BASE_DIR = Path(__file__).resolve().parent
MAX_SUBJECT = 200
MAX_MESSAGE = 5000
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")
# A pre-computed hash so login takes similar time whether or not the email exists.
_DUMMY_HASH = generate_password_hash("spamguard-dummy-password")


def _load_secret_key(instance_dir: Path) -> str:
    key = os.environ.get("SECRET_KEY")
    if key:
        return key
    # Development fallback: generate once and keep in the (git-ignored) instance folder.
    instance_dir.mkdir(parents=True, exist_ok=True)
    key_file = instance_dir / "secret_key"
    if not key_file.exists():
        key_file.write_text(secrets.token_hex(32))
        logging.getLogger("spamguard").warning(
            "SECRET_KEY not set; generated a local one in %s. Set SECRET_KEY in .env for deployment.", key_file)
    return key_file.read_text().strip()


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__, instance_path=str(BASE_DIR / "instance"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app.config.update(
        SECRET_KEY=_load_secret_key(Path(app.instance_path)),
        DATABASE_PATH=os.environ.get("DATABASE_PATH", str(Path(app.instance_path) / "spamguard.db")),
        DATASET_PATH=os.environ.get("DATASET_PATH", str(BASE_DIR / "spam.csv")),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "0") == "1",
        PERMANENT_SESSION_LIFETIME=timedelta(days=14),
        MAX_CONTENT_LENGTH=64 * 1024,
        RATELIMIT_ENABLED=True,
    )
    if test_config:
        app.config.update(test_config)

    init_db(app.config["DATABASE_PATH"])
    app.teardown_appcontext(close_db)
    spam_model = SpamModel(app.config["DATASET_PATH"])
    app.extensions["spam_model"] = spam_model

    # ------------------------------------------------------- rate limiting
    hits: dict[tuple, collections.deque] = collections.defaultdict(collections.deque)

    def rate_limited(bucket: str, limit: int, window: int = 60) -> bool:
        if not app.config["RATELIMIT_ENABLED"]:
            return False
        now = time.monotonic()
        q = hits[(bucket, request.remote_addr)]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            return True
        q.append(now)
        return False

    # ----------------------------------------------------------- security
    def csrf_token() -> str:
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(32)
        return session["csrf"]

    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.before_request
    def _before():
        g.user = None
        uid = session.get("uid")
        if uid:
            row = get_db().execute("SELECT id, name, email FROM users WHERE id = ?", (uid,)).fetchone()
            if row:
                g.user = row
            else:
                session.pop("uid", None)
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            sent = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token", "")
            expected = session.get("csrf", "")
            if not expected or not hmac.compare_digest(str(sent), expected):
                if request.path.startswith("/api/") or request.path == "/predict":
                    return jsonify(error="Security token missing or expired. Refresh the page and try again."), 403
                abort(400, "Security token missing or expired. Please go back, refresh the page and try again.")

    @app.after_request
    def _headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; img-src 'self' data:; "
            "style-src 'self' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
        if request.path.startswith("/api/") or request.path in ("/predict", "/history/export.csv"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    # -------------------------------------------------------------- helpers
    def safe_next(target: str | None) -> str:
        if not target:
            return url_for("dashboard")
        p = urlparse(target)
        if p.scheme or p.netloc or not target.startswith("/") or target.startswith("//"):
            return url_for("dashboard")
        return target

    def owner_clause(create_guest: bool = False):
        """SQL fragment + params restricting scans to the current visitor."""
        if g.user:
            return "user_id = ?", (g.user["id"],)
        if create_guest and "gid" not in session:
            session["gid"] = secrets.token_hex(16)
        gid = session.get("gid")
        if gid:
            return "guest_id = ?", (gid,)
        return "0 = 1", ()

    def json_error(msg: str, code: int):
        return jsonify(error=msg), code

    def scan_to_dict(row) -> dict:
        return {"id": row["id"], "subject": row["subject"], "message": row["message"],
                "label": row["label"], "spam_score": row["spam_score"],
                "created_at": row["created_at"]}

    # ---------------------------------------------------------------- pages
    @app.context_processor
    def _inject():
        return {"model_metrics": spam_model.metrics, "year": datetime.now().year}

    @app.route("/")
    def home():
        return render_template("index.html", max_message=MAX_MESSAGE, max_subject=MAX_SUBJECT)

    @app.route("/analyze")
    def analyze_page():
        return render_template("analyze.html", max_message=MAX_MESSAGE, max_subject=MAX_SUBJECT)

    @app.route("/dashboard")
    def dashboard():
        return render_template("dashboard.html")

    @app.route("/history")
    def history_page():
        return render_template("history.html")

    @app.route("/evaluation")
    def evaluation():
        return render_template("evaluation.html", m=spam_model.metrics)

    # ----------------------------------------------------------------- auth
    def adopt_guest_scans(guest_id: str | None, user_id: int) -> None:
        if guest_id:
            db = get_db()
            db.execute("UPDATE scans SET user_id = ?, guest_id = NULL WHERE guest_id = ?", (user_id, guest_id))
            db.commit()

    def start_session(user_id: int) -> None:
        gid = session.get("gid")
        session.clear()
        session.permanent = True
        session["uid"] = user_id
        csrf_token()
        adopt_guest_scans(gid, user_id)

    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        if g.user:
            return redirect(url_for("dashboard"))
        form = {"name": "", "email": ""}
        if request.method == "POST":
            if rate_limited("auth", 10):
                flash("Too many attempts. Please wait a minute and try again.", "error")
                return render_template("signup.html", form=form), 429
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            confirm = request.form.get("confirm", "")
            form = {"name": name, "email": email}
            errors = []
            if not 2 <= len(name) <= 60:
                errors.append("Name must be between 2 and 60 characters.")
            if not EMAIL_RE.match(email) or len(email) > 254:
                errors.append("Enter a valid email address.")
            if not 8 <= len(password) <= 128:
                errors.append("Password must be 8 to 128 characters long.")
            elif password.isdigit() or password.lower() == password and password.isalpha():
                errors.append("Password should mix letters with numbers or symbols.")
            if password != confirm:
                errors.append("Passwords do not match.")
            if not errors:
                try:
                    db = get_db()
                    cur = db.execute("INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                                     (name, email, generate_password_hash(password)))
                    db.commit()
                except sqlite3.IntegrityError:
                    errors.append("An account with that email already exists.")
                else:
                    start_session(cur.lastrowid)
                    flash("Welcome to SpamGuard! Your account is ready.", "success")
                    return redirect(url_for("dashboard"))
            for e in errors:
                flash(e, "error")
            return render_template("signup.html", form=form), 400
        return render_template("signup.html", form=form)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if g.user:
            return redirect(safe_next(request.args.get("next")))
        nxt = request.values.get("next", "")
        if request.method == "POST":
            if rate_limited("auth", 10):
                flash("Too many attempts. Please wait a minute and try again.", "error")
                return render_template("login.html", next=nxt, email=""), 429
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            row = get_db().execute("SELECT id, password_hash FROM users WHERE email = ?", (email,)).fetchone()
            ok = check_password_hash(row["password_hash"] if row else _DUMMY_HASH, password)
            if row and ok:
                start_session(row["id"])
                return redirect(safe_next(nxt))
            flash("Incorrect email or password.", "error")
            return render_template("login.html", next=nxt, email=email), 401
        return render_template("login.html", next=nxt, email="")

    @app.post("/logout")
    def logout():
        session.clear()
        flash("You have been signed out.", "success")
        return redirect(url_for("home"))

    # ------------------------------------------------------------------ API
    @app.post("/api/predict")
    @app.post("/predict")  # path kept from the original project
    def predict():
        if rate_limited("predict", 60):
            return json_error("Too many requests. Please slow down.", 429)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return json_error("Send a JSON body with 'subject' and/or 'message'.", 400)
        subject, message = payload.get("subject", ""), payload.get("message", "")
        if not isinstance(subject, str) or not isinstance(message, str):
            return json_error("'subject' and 'message' must be text.", 400)
        subject, message = subject.strip(), message.strip()
        if not subject and not message:
            return json_error("Please enter an email subject or message.", 400)
        if len(subject) > MAX_SUBJECT:
            return json_error(f"Subject is too long (max {MAX_SUBJECT} characters).", 400)
        if len(message) > MAX_MESSAGE:
            return json_error(f"Message is too long (max {MAX_MESSAGE} characters).", 400)

        text = f"{subject} {message}".strip()
        res = spam_model.predict(text)
        signals = analyze_signals(text)
        is_spam = res["prediction"] == "spam"
        score = res["spam_score"]
        risk = "low" if score < 25 else "medium" if score < 50 else "high" if score < 75 else "critical"

        saved_id = None
        if payload.get("save", True):
            clause, _ = owner_clause(create_guest=True)
            db = get_db()
            cur = db.execute(
                "INSERT INTO scans (user_id, guest_id, subject, message, label, spam_score, terms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (g.user["id"] if g.user else None, None if g.user else session["gid"], subject, message,
                 res["prediction"], score, json.dumps([t["word"] for t in res["spam_terms"]])))
            db.commit()
            saved_id = cur.lastrowid

        return jsonify({
            "id": saved_id,
            "prediction": res["prediction"],
            "result": "SPAM" if is_spam else "HAM",
            "spam_score": score,
            "risk": risk,
            "description": ("This message was classified as potentially unwanted or suspicious."
                            if is_spam else "This message was classified as a normal message."),
            "spam_terms": res["spam_terms"],
            "ham_terms": res["ham_terms"],
            "recognized_terms": res["recognized_terms"],
            "signals": signals,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

    @app.get("/api/history")
    def api_history():
        try:
            page = max(1, int(request.args.get("page", 1)))
            per = min(50, max(1, int(request.args.get("per_page", 10))))
        except ValueError:
            return json_error("Invalid pagination values.", 400)
        label = request.args.get("label", "")
        q = request.args.get("q", "").strip()[:100]
        clause, params = owner_clause()
        where, args = [clause], list(params)
        if label in ("spam", "ham"):
            where.append("label = ?")
            args.append(label)
        if q:
            esc = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            where.append("(subject LIKE ? ESCAPE '\\' OR message LIKE ? ESCAPE '\\')")
            args += [f"%{esc}%", f"%{esc}%"]
        sql_where = " AND ".join(where)
        db = get_db()
        total = db.execute(f"SELECT COUNT(*) FROM scans WHERE {sql_where}", args).fetchone()[0]
        rows = db.execute(
            f"SELECT * FROM scans WHERE {sql_where} ORDER BY id DESC LIMIT ? OFFSET ?",
            args + [per, (page - 1) * per]).fetchall()
        return jsonify({"items": [scan_to_dict(r) for r in rows], "total": total,
                        "page": page, "per_page": per, "pages": max(1, -(-total // per))})

    @app.delete("/api/history/<int:scan_id>")
    def api_history_delete(scan_id: int):
        clause, params = owner_clause()
        db = get_db()
        cur = db.execute(f"DELETE FROM scans WHERE id = ? AND {clause}", (scan_id, *params))
        db.commit()
        if cur.rowcount == 0:
            return json_error("Scan not found.", 404)
        return jsonify(ok=True)

    @app.delete("/api/history")
    def api_history_clear():
        clause, params = owner_clause()
        db = get_db()
        cur = db.execute(f"DELETE FROM scans WHERE {clause}", params)
        db.commit()
        return jsonify(ok=True, deleted=cur.rowcount)

    @app.get("/history/export.csv")
    def export_csv():
        clause, params = owner_clause()
        rows = get_db().execute(
            f"SELECT created_at, label, spam_score, subject, message FROM scans WHERE {clause} ORDER BY id DESC",
            params).fetchall()

        def safe(v):  # neutralise spreadsheet formula injection
            v = str(v)
            return "'" + v if v[:1] in ("=", "+", "-", "@", "\t", "\r") else v

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["scanned_at_utc", "verdict", "spam_score_percent", "subject", "message"])
        for r in rows:
            w.writerow([r["created_at"], r["label"], r["spam_score"], safe(r["subject"]), safe(r["message"])])
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=spamguard-history.csv"})

    @app.get("/api/stats")
    def api_stats():
        clause, params = owner_clause()
        db = get_db()
        t = db.execute(
            f"SELECT COUNT(*) n, COALESCE(SUM(label='spam'),0) spam, COALESCE(SUM(label='ham'),0) ham, "
            f"AVG(spam_score) avg_score, MAX(created_at) last_at FROM scans WHERE {clause}", params).fetchone()
        today = datetime.now(timezone.utc).date()
        days = [(today - timedelta(days=i)).isoformat() for i in range(13, -1, -1)]
        per_day = {r["d"]: r for r in db.execute(
            f"SELECT substr(created_at,1,10) d, COUNT(*) n, SUM(label='spam') spam FROM scans "
            f"WHERE {clause} AND substr(created_at,1,10) >= ? GROUP BY d", (*params, days[0])).fetchall()}
        timeline = [{"date": d, "total": per_day[d]["n"] if d in per_day else 0,
                     "spam": (per_day[d]["spam"] or 0) if d in per_day else 0} for d in days]
        buckets = [0] * 5
        for r in db.execute(f"SELECT spam_score FROM scans WHERE {clause}", params):
            buckets[min(4, int(r["spam_score"] // 20))] += 1
        counter: collections.Counter = collections.Counter()
        for r in db.execute(f"SELECT terms FROM scans WHERE {clause} AND label = 'spam' ORDER BY id DESC LIMIT 500", params):
            try:
                counter.update(json.loads(r["terms"]))
            except ValueError:
                pass
        recent = db.execute(
            f"SELECT * FROM scans WHERE {clause} ORDER BY id DESC LIMIT 6", params).fetchall()
        total = t["n"]
        return jsonify({
            "total": total, "spam": t["spam"], "ham": t["ham"],
            "spam_rate": round(t["spam"] / total * 100, 1) if total else 0,
            "avg_score": round(t["avg_score"], 1) if total else 0,
            "last_scan": t["last_at"],
            "timeline": timeline,
            "score_buckets": [{"range": f"{i*20}–{i*20+20}%", "count": c} for i, c in enumerate(buckets)],
            "top_terms": [{"word": w, "count": c} for w, c in counter.most_common(8)],
            "recent": [scan_to_dict(r) for r in recent],
            "authenticated": bool(g.user),
        })

    @app.get("/api/health")
    def health():
        get_db().execute("SELECT 1").fetchone()
        return jsonify(status="ok", model="MultinomialNB + TF-IDF", accuracy=spam_model.metrics["accuracy"])

    # --------------------------------------------------------------- errors
    def wants_json() -> bool:
        return request.path.startswith("/api/") or request.path == "/predict"

    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(405)
    @app.errorhandler(413)
    @app.errorhandler(500)
    def _error(err):
        code = getattr(err, "code", 500) or 500
        if code == 500:
            logging.getLogger("spamguard").exception("Unhandled error", exc_info=err)
        titles = {400: "Bad request", 404: "Page not found", 405: "Method not allowed",
                  413: "Request too large", 500: "Something went wrong"}
        msg = (getattr(err, "description", "") if code in (400, 413) else "") or {
            404: "The page you are looking for doesn't exist or has moved.",
            405: "That action isn't allowed on this page.",
            500: "An unexpected error occurred. Please try again."}.get(code, "")
        if wants_json():
            return jsonify(error=msg or titles.get(code, "Error")), code
        return render_template("error.html", code=code, title=titles.get(code, "Error"), message=msg), code

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG", "0") == "1")
