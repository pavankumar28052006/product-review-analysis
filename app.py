from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
)
from werkzeug.security import generate_password_hash, check_password_hash

import joblib

from database import get_connection, init_db

MODEL_PATH = Path("models") / "sentiment_svm.pkl"


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "change-this-secret-key-for-production"

    init_db()

    if MODEL_PATH.exists():
        app.sentiment_model = joblib.load(MODEL_PATH)
    else:
        app.sentiment_model = None
        print(
            f"WARNING: Model file {MODEL_PATH} not found. "
            "Train the model first by running: python sentiment_model.py"
        )



    def get_current_user():
        user_id = session.get("user_id")
        if not user_id:
            return None
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cur.fetchone()
        conn.close()
        return user

    def login_required(view_func):
        def wrapped(*args, **kwargs):
            if not get_current_user():
                flash("Please log in to access this page.", "warning")
                return redirect(url_for("login"))
            return view_func(*args, **kwargs)

        wrapped.__name__ = view_func.__name__
        return wrapped



    @app.route("/", methods=["GET", "POST"])
    @app.route("/dashboard", methods=["GET", "POST"])
    @login_required
    def dashboard():
        user = get_current_user()
        prediction = None
        review_text = ""

        if request.method == "POST":
            review_text = request.form.get("review_text", "").strip()
            if not review_text:
                flash("Please enter a review text.", "danger")
            elif not app.sentiment_model:
                flash(
                    "Sentiment model is not loaded. "
                    "Train it first using 'python sentiment_model.py'.",
                    "danger",
                )
            else:
                prediction = app.sentiment_model.predict([review_text])[0]

                conn = get_connection()
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO reviews (user_id, review_text, predicted_sentiment)
                    VALUES (?, ?, ?)
                    """,
                    (user["id"], review_text, prediction),
                )
                conn.commit()
                conn.close()

        return render_template(
            "dashboard.html",
            user=user,
            prediction=prediction,
            review_text=review_text,
        )

    @app.route("/history")
    @login_required
    def history():
        user = get_current_user()
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT review_text, predicted_sentiment, created_at
            FROM reviews
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user["id"],),
        )
        reviews = cur.fetchall()
        conn.close()
        return render_template("history.html", user=user, reviews=reviews)

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            email = request.form.get("email", "").strip()
            password = request.form.get("password", "").strip()
            confirm_password = request.form.get("confirm_password", "").strip()

            if not username or not email or not password:
                flash("All fields are required.", "danger")
            elif password != confirm_password:
                flash("Passwords do not match.", "danger")
            else:
                password_hash = generate_password_hash(password)
                conn = get_connection()
                cur = conn.cursor()
                try:
                    cur.execute(
                        """
                        INSERT INTO users (username, email, password_hash)
                        VALUES (?, ?, ?)
                        """,
                        (username, email, password_hash),
                    )
                    conn.commit()
                    flash("Registration successful. Please log in.", "success")
                    return redirect(url_for("login"))
                except Exception:
                    flash(
                        "Username or email already exists. Please choose another.",
                        "danger",
                    )
                finally:
                    conn.close()

        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip()
            password = request.form.get("password", "").strip()

            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE email = ?", (email,))
            user = cur.fetchone()
            conn.close()

            if user and check_password_hash(user["password_hash"], password):
                session["user_id"] = user["id"]
                flash("Logged in successfully.", "success")
                return redirect(url_for("dashboard"))
            else:
                flash("Invalid email or password.", "danger")

        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        flash("You have been logged out.", "info")
        return redirect(url_for("login"))

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(debug=True)


