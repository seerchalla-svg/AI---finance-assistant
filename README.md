# 💰 Wisp — AI Finance Assistant

A beginner-friendly personal finance tracker that uses Machine Learning to automatically categorize expenses, track budgets, and generate simple spending insights — built as a portfolio project to demonstrate practical AI/ML integration in a real-world full-stack application.

> ⚠️ This project is for **educational and portfolio purposes only**. It does not provide professional investment, tax, or financial advice.

---

## 📌 Overview

Manually tracking expenses and figuring out where your money goes is tedious. Wisp solves that by combining a standard finance tracker with a trained ML model:

- Enter a transaction like `"swiggy dinner 450"` → a trained classifier predicts the category (**Food**) automatically
- Correct a wrong prediction with one click — corrections are logged for future improvement
- Track income, expenses, and balance on a live dashboard
- Set monthly budgets per category and get alerted when you're close to a limit
- See month-over-month spending patterns and a simple linear forecast for next month
- Ask a built-in chatbot about your budgets, forecast, or where to save — answered using your real account data

---

## ✨ Features

- 🔐 **Authentication** — register/login with hashed passwords (Werkzeug), session persisted via localStorage
- 💸 **Transaction management** — add income/expense, edit, delete, full transaction list
- 🤖 **AI expense categorization** — TF-IDF + trained classifier predicts a category from the description; user can correct it, and corrections are stored
- 📊 **Budget tracking** — set a monthly limit per category, live progress bars, alert banners at 80%+ usage or over-budget
- 📈 **Dashboard** — income/expense/balance stat cards + a monthly spending trend chart (Chart.js)
- 🔁 **Spending patterns** — month-over-month category comparison (up/down/flat with % change)
- 🔮 **Forecast** — next month's estimated spend using linear regression over historical monthly totals
- 💬 **Finance chatbot** — rule-based assistant that answers questions about your budgets, forecast, and top spending category using your actual data
- ⚠️ Every insight/forecast screen includes a disclaimer that this isn't professional financial advice

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, vanilla JavaScript, Chart.js |
| Backend | Python, Flask, Flask-CORS |
| Database | SQLite via Flask-SQLAlchemy |
| Auth | Werkzeug password hashing |
| Machine Learning | scikit-learn (TF-IDF + classifier), joblib for model persistence |
| Data Processing | pandas, NumPy |

---

## 🤖 Machine Learning

**Expense Classification**
- Input: transaction description
- Vectorized with a saved `TfidfVectorizer`
- Classified using a trained model (`expense_classifier.pkl`) into categories like Food, Transport, Shopping, Bills, etc.
- Trained in `ml/train_model.ipynb` on `data/transactions_clean.csv`
- User corrections are logged to a `Correction` table for potential retraining later

**Spending Forecast**
- Pulls monthly expense totals from the database
- Fits a simple linear trend (`numpy.polyfit`) across months to estimate next month's spend
- Falls back to a plain estimate when there's less than 2 months of data

---

## 📂 Project Structure
wisp-app/
├── backend/
│   ├── app.py              # Flask API — auth, transactions, budgets, ML prediction, chatbot, forecast
│   ├── text.html            # raw HTML form used to test each API endpoint directly (dev/debug only)
│   └── finance.db           # SQLite database (auto-created on first run)
├── frontend/
│   └── index.html            # the actual Wisp UI (auth, dashboard, budgets, insights, chatbot)
├── ml/
│   ├── train_model.ipynb     # model training notebook
│   ├── expense_classifier.pkl
│   └── tfidf_vectorizer.pkl
├── data/
│   └── transactions_clean.csv
├── screenshots/
└── README.md

## 🖼️ Screenshots

📄  [Screenshots.pdf](https://github.com/user-attachments/files/32437369/Screenshots.pdf)


---

## 🚀 Getting Started

```bash
# clone the repo
git clone https://github.com/<your-username>/wisp-app.git
cd wisp-app

# backend setup
cd backend
pip install flask flask-sqlalchemy flask-cors werkzeug joblib numpy
python app.py
# server runs at http://127.0.0.1:5000
Then open frontend/index.html directly in your browser (it calls the API at http://127.0.0.1:5000, so keep the backend running).
Note: expense_classifier.pkl and tfidf_vectorizer.pkl must exist in ml/ before starting the backend — run ml/train_model.ipynb first if they're missing.

## 🔮 Future Enhancements

- Move the chatbot from rule-based responses to an actual NLP/LLM-backed model
- Search and filter on the Transactions page
- Receipt scanning with OCR
- Bank/payment-provider integrations
- Mobile app version
- Cloud deployment

---

## ⚠️ Limitations

- Category predictions depend on the quality/quantity of training data in `transactions_clean.csv`
- Forecast needs at least 2 months of transaction history to be meaningful
- Chatbot responses are keyword-matched, not a trained language model
- Not a substitute for a qualified financial advisor

---

## 👩‍💻 Author

**Seershika** — B.Tech Chemical Engineering (Minor: Applied Geophysics), IIT Dhanbad[Screenshots.pdf](https://github.com/user-attachments/files/32437498/Screenshots.pdf)

