# 🚀 CredTech – Explainable Credit Scorecard

An end-to-end, **real-time explainable credit intelligence** mini-platform that:
- Ingests **structured** (market data) and **unstructured** (news) signals,
- Generates a **credit risk score** per issuer,
- Explains **why** via feature contributions and trends,
- Presents everything in an interactive **Streamlit** dashboard.

**Live Demo:** <!-- Replace with your deployed URL -->
https://YOUR-APP-NAME.streamlit.app


## 1) Problem → Our Scope 

**Goal:** Build an explainable credit scoring system that updates frequently, combines structured + unstructured data, and shows **clear explanations** in an analyst-friendly dashboard.  
- **Structured data:** Yahoo Finance OHLCV prices (v1).  
- **Unstructured data:** Yahoo Finance RSS headlines with sentiment tagging (v1).  
- **Scoring:** Binary classifier (Safe/Risky) + probability score.  
- **Explainability:** SHAP feature importance + concise text reasons.  
- **Dashboard:** Trends, filters, badges, explanations, news sentiment.

> Roadmap to fully match the brief: add a second structured source (e.g., World Bank, FRED, or Alpha Vantage) and enrich unstructured event mapping.


## 2) System Architecture (High-Level)

**Ingestion → Features → Model → Explainability → Dashboard → Deploy**

- **Data Ingestion (src/data_ingestion.py):**
  - Pulls daily OHLCV for selected tickers via `yfinance`.
  - Fetches news headlines via Yahoo Finance RSS (best-effort; resilient to failures).
  - Saves CSVs under `data/`.
- **Feature Engineering (src/model.py):**
  - Return/volatility factors, rolling stats, volume signals.
  - Train/test split + class-imbalance handling (SMOTE).
  - Model: `RandomForestClassifier` with tuned params.
  - Saves SHAP importance plot to `data/shap_feature_importance.png`.
- **Explainability:**
  - Global feature importance via SHAP.
  - Human-readable interpretation strings (e.g., “High short-term volatility ↑ risk”).
- **Dashboard (dashboard/app.py):**
  - Ticker selector, risk score badge (Safe/Risky), probability gauge, trend line.
  - SHAP bar plot + short textual explanation.
  - Latest headlines + sentiment tags.
- **Deployment:**
  - Streamlit Cloud (public demo link).
  - Optional: `.streamlit/config.toml` for dark theme.

## 3) Repository Structure

CredTech-Credit-Scorecard/
│
├─ data/
│ ├─ stock_data.csv # Structured input (OHLCV)
│ ├─ news_data.csv # Unstructured input (RSS headlines)
│ └─ shap_feature_importance.png
│
├─ src/
│ ├─ data_ingestion.py # Fetch + save structured/unstructured data
│ ├─ model.py # Features, training, SHAP, metrics
│ 
│
├─ dashboard/
│ └─ app.py # Streamlit UI
│
├─ requirements.txt
├─ README.md # (this file)

### 4. Solution Approach 🚀
Collect stock data + news data (Yahoo Finance API).

Clean data → generate features (returns, volatility).

Train ML model (RandomForest + SMOTE) to predict credit risk.

Use SHAP for explainability (which features drive risk).

### 5. Architecture ⚙️

Data → Preprocessing → Model Training → Explainability → Dashboard

Data: Stock prices + News headlines

Model: RandomForest Classifier

Explainability: SHAP feature importance

Frontend: Streamlit dashboard

### 6. Results 📊
Accuracy: ~89% (RandomForest + SMOTE)

Balanced precision & recall

SHAP shows top drivers of credit risk (volatility, returns, volume)

### 7. Deployment 🌐

Live Streamlit App: 👉 https://credtech-credit-scorecard-g48vw8cpcmwo6jcay8uugv.streamlit.app/

Judges can interact with:

Choose a stock ticker

See credit score (Safe / Risky)

View feature importance + news impact

### 8. How to Run Locally 💻

git clone https://github.com/Saipriya37/CredTech-Credit-Scorecard.git
cd CredTech-Credit-Scorecard
pip install -r requirements.txt
streamlit run dashboard/app.py



