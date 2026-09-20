from flask import Flask, request, jsonify
import joblib
import os
from datetime import datetime
from collections import defaultdict
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "finance.db")
db = SQLAlchemy(app)

# ---------------- TABLES ----------------

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    description = db.Column(db.String(200), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    transaction_type = db.Column(db.String(10), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Budget(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    limit_amount = db.Column(db.Float, nullable=False)

class Correction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(200))
    corrected_category = db.Column(db.String(50))

with app.app_context():
    db.create_all()

model = joblib.load(os.path.join(BASE_DIR, "..", "ml", "expense_classifier.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "..", "ml", "tfidf_vectorizer.pkl"))

# ---------------- BASIC ----------------

@app.route("/")
def home():
    return "AI Finance Assistant backend is running!"

# ---------------- AUTH ----------------

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    name = data.get("name"); email = data.get("email"); password = data.get("password")
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 400
    u = User(name=name, email=email, password_hash=generate_password_hash(password))
    db.session.add(u); db.session.commit()
    return jsonify({"message": "Registered successfully", "user_id": u.id})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email"); password = data.get("password")
    u = User.query.filter_by(email=email).first()
    if not u or not check_password_hash(u.password_hash, password):
        return jsonify({"error": "Invalid email or password"}), 401
    return jsonify({"message": "Login successful", "user_id": u.id, "name": u.name})

# ---------------- TRANSACTIONS ----------------

@app.route("/predict-category", methods=["POST"])
def predict_category():
    data = request.get_json()
    user_id = data.get("user_id"); description = data.get("description", ""); amount = data.get("amount", 0)
    vec = vectorizer.transform([description])
    prediction = model.predict(vec)[0]
    t = Transaction(user_id=user_id, description=description, amount=amount,
                     category=prediction, transaction_type="expense")
    db.session.add(t); db.session.commit()
    return jsonify({"category": prediction, "saved": True, "transaction_id": t.id})

@app.route("/add-income", methods=["POST"])
def add_income():
    data = request.get_json()
    user_id = data.get("user_id"); description = data.get("description", ""); amount = data.get("amount", 0)
    t = Transaction(user_id=user_id, description=description, amount=amount,
                     category="Income", transaction_type="income")
    db.session.add(t); db.session.commit()
    return jsonify({"saved": True, "transaction_id": t.id})

@app.route("/correct-category", methods=["POST"])
def correct_category():
    data = request.get_json()
    t = Transaction.query.get(data.get("transaction_id"))
    if not t:
        return jsonify({"error": "not found"}), 404
    t.category = data.get("category")
    db.session.add(Correction(description=t.description, corrected_category=t.category))
    db.session.commit()
    return jsonify({"message": "corrected"})

@app.route("/transactions", methods=["GET"])
def get_transactions():
    user_id = request.args.get("user_id")
    txns = Transaction.query.filter_by(user_id=user_id).all()
    return jsonify([{"id": t.id, "description": t.description, "amount": t.amount,
                      "category": t.category, "transaction_type": t.transaction_type} for t in txns])

@app.route("/transactions/<int:id>", methods=["PUT"])
def update_transaction(id):
    t = Transaction.query.get(id)
    if not t:
        return jsonify({"error": "Transaction not found"}), 404
    data = request.get_json()
    t.description = data.get("description", t.description)
    t.amount = data.get("amount", t.amount)
    t.category = data.get("category", t.category)
    t.transaction_type = data.get("transaction_type", t.transaction_type)
    db.session.commit()
    return jsonify({"message": "Updated successfully"})

@app.route("/transactions/<int:id>", methods=["DELETE"])
def delete_transaction(id):
    t = Transaction.query.get(id)
    if not t:
        return jsonify({"error": "Transaction not found"}), 404
    db.session.delete(t); db.session.commit()
    return jsonify({"message": "Deleted successfully"})

# ---------------- BUDGETS ----------------

@app.route("/budgets", methods=["POST"])
def create_budget():
    data = request.get_json()
    user_id = data.get("user_id"); category = data.get("category"); limit_amount = data.get("limit_amount")
    existing = Budget.query.filter(Budget.user_id == user_id,
                                    db.func.lower(Budget.category) == db.func.lower(category)).first()
    if existing:
        existing.limit_amount = limit_amount; db.session.commit()
        return jsonify({"message": "Budget updated", "id": existing.id})
    b = Budget(user_id=user_id, category=category, limit_amount=limit_amount)
    db.session.add(b); db.session.commit()
    return jsonify({"message": "Budget created", "id": b.id})

@app.route("/budgets", methods=["GET"])
def get_budgets():
    user_id = request.args.get("user_id")
    budgets = Budget.query.filter_by(user_id=user_id).all()
    result = []
    for b in budgets:
        spent = db.session.query(db.func.sum(Transaction.amount)).filter(
            Transaction.user_id == user_id,
            db.func.lower(Transaction.category) == db.func.lower(b.category),
            Transaction.transaction_type == "expense"
        ).scalar() or 0
        result.append({"id": b.id, "category": b.category, "limit_amount": b.limit_amount, "spent": spent})
    return jsonify(result)

# ---------------- DASHBOARD / TREND / FORECAST / INSIGHTS ----------------

@app.route("/dashboard", methods=["GET"])
def get_dashboard():
    user_id = request.args.get("user_id")
    total_income = db.session.query(db.func.sum(Transaction.amount)).filter(
        Transaction.user_id == user_id, Transaction.transaction_type == "income").scalar() or 0
    total_expenses = db.session.query(db.func.sum(Transaction.amount)).filter(
        Transaction.user_id == user_id, Transaction.transaction_type == "expense").scalar() or 0
    breakdown = db.session.query(Transaction.category, db.func.sum(Transaction.amount)).filter(
        Transaction.user_id == user_id, Transaction.transaction_type == "expense").group_by(Transaction.category).all()
    return jsonify({
        "total_income": total_income, "total_expenses": total_expenses,
        "balance": total_income - total_expenses,
        "category_breakdown": [{"category": c, "total": t} for c, t in breakdown]
    })

def _monthly_totals(user_id):
    txns = Transaction.query.filter_by(user_id=user_id, transaction_type="expense").all()
    monthly = defaultdict(float)
    for t in txns:
        key = t.created_at.strftime("%Y-%m") if t.created_at else "Unknown"
        monthly[key] += t.amount
    months = sorted(monthly.keys())
    return months, [monthly[m] for m in months]

@app.route("/monthly-trend", methods=["GET"])
def get_monthly_trend():
    user_id = request.args.get("user_id")
    months, totals = _monthly_totals(user_id)
    return jsonify({"months": months, "totals": totals})

@app.route("/forecast", methods=["GET"])
def get_forecast():
    user_id = request.args.get("user_id")
    months, totals = _monthly_totals(user_id)
    if len(totals) < 2:
        estimate = totals[0] if totals else 0
        return jsonify({"estimate": round(estimate, 2),
                         "note": "Not enough historical data yet — showing current total as a rough estimate."})
    x = np.arange(len(totals)); y = np.array(totals)
    coeffs = np.polyfit(x, y, 1)
    estimate = max(0, coeffs[0] * len(totals) + coeffs[1])
    return jsonify({"estimate": round(estimate, 2), "note": "Estimate based on your spending trend. Not a guarantee."})

@app.route("/insights", methods=["GET"])
def get_insights():
    user_id = request.args.get("user_id")
    insights = []
    category_totals = db.session.query(Transaction.category, db.func.sum(Transaction.amount)).filter(
        Transaction.user_id == user_id, Transaction.transaction_type == "expense").group_by(Transaction.category).all()
    if category_totals:
        top_category, top_amount = max(category_totals, key=lambda x: x[1])
        insights.append(f"Your highest spending category is {top_category} at \u20b9{top_amount:.2f}.")
    for b in Budget.query.filter_by(user_id=user_id).all():
        spent = db.session.query(db.func.sum(Transaction.amount)).filter(
            Transaction.user_id == user_id,
            db.func.lower(Transaction.category) == db.func.lower(b.category),
            Transaction.transaction_type == "expense"
        ).scalar() or 0
        if spent > b.limit_amount:
            insights.append(f"You've exceeded your {b.category} budget by \u20b9{spent - b.limit_amount:.2f}.")
        elif spent > 0.8 * b.limit_amount:
            insights.append(f"You're close to your {b.category} budget limit \u2014 {spent:.2f} of {b.limit_amount:.2f} used.")
    if not insights:
        insights.append("Not enough data yet for insights. Add a few transactions first.")
    insights.append("These insights are educational and not professional financial advice.")
    return jsonify({"insights": insights})

# ---------------- PATTERNS (month-over-month category comparison) ----------------

@app.route("/patterns", methods=["GET"])
def get_patterns():
    user_id = request.args.get("user_id")
    txns = Transaction.query.filter_by(user_id=user_id, transaction_type="expense").all()
    monthly_cat = defaultdict(lambda: defaultdict(float))
    for t in txns:
        month = t.created_at.strftime("%Y-%m") if t.created_at else "Unknown"
        monthly_cat[month][t.category] += t.amount
    months = sorted(monthly_cat.keys())
    if len(months) < 2:
        return jsonify({"patterns": [], "note": "Need at least two months of data to detect patterns yet."})
    current, previous = months[-1], months[-2]
    cats = set(monthly_cat[current]) | set(monthly_cat[previous])
    patterns = []
    for c in cats:
        cur = monthly_cat[current].get(c, 0)
        prev = monthly_cat[previous].get(c, 0)
        change = 100.0 if prev == 0 and cur > 0 else (0.0 if prev == 0 else round((cur - prev) / prev * 100, 1))
        direction = "up" if change > 5 else ("down" if change < -5 else "flat")
        patterns.append({"category": c, "current": cur, "previous": prev, "change_pct": change, "direction": direction})
    patterns.sort(key=lambda x: -abs(x["change_pct"]))
    return jsonify({"patterns": patterns, "note": f"Comparing {current} to {previous}"})

# ---------------- CHATBOT (rule-based, uses real account data) ----------------

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_id = data.get("user_id")
    message = (data.get("message") or "").lower()

    if "budget" in message:
        budgets = Budget.query.filter_by(user_id=user_id).all()
        if not budgets:
            return jsonify({"reply": "You haven't set any budgets yet. Go to Budgets to create one."})
        lines = []
        for b in budgets:
            spent = db.session.query(db.func.sum(Transaction.amount)).filter(
                Transaction.user_id == user_id,
                db.func.lower(Transaction.category) == db.func.lower(b.category),
                Transaction.transaction_type == "expense"
            ).scalar() or 0
            lines.append(f"{b.category}: \u20b9{spent:.0f} of \u20b9{b.limit_amount:.0f}")
        return jsonify({"reply": "Here's your budget status:\n" + "\n".join(lines)})

    if "forecast" in message or "next month" in message:
        months, totals = _monthly_totals(user_id)
        if len(totals) < 2:
            est = totals[0] if totals else 0
        else:
            x = np.arange(len(totals)); y = np.array(totals)
            coeffs = np.polyfit(x, y, 1)
            est = max(0, coeffs[0] * len(totals) + coeffs[1])
        return jsonify({"reply": f"Based on your trend, you're estimated to spend about \u20b9{est:.0f} next month."})

    if "save" in message or "reduce" in message or "spend less" in message:
        category_totals = db.session.query(Transaction.category, db.func.sum(Transaction.amount)).filter(
            Transaction.user_id == user_id, Transaction.transaction_type == "expense").group_by(Transaction.category).all()
        if not category_totals:
            return jsonify({"reply": "Add a few transactions first, then I can suggest where to save."})
        top_cat, top_amt = max(category_totals, key=lambda x: x[1])
        return jsonify({"reply": f"Your highest spending category is {top_cat} at \u20b9{top_amt:.0f}. That's the best place to look for savings."})

    if "categor" in message and ("how" in message or "predict" in message or "work" in message):
        return jsonify({"reply": "I use a machine learning model trained on transaction descriptions to guess a category like Food or Transport. If I get it wrong, you can correct it on the Add Transaction page."})

    if "hi" in message or "hello" in message or "hey" in message:
        return jsonify({"reply": "Hi! Ask me about your budgets, spending forecast, or where you could save."})

    return jsonify({"reply": "I can help with budgets, spending forecasts, and savings tips. Try asking 'How is my food budget?' or 'What's my forecast?'"})

if __name__ == "__main__":
    app.run()