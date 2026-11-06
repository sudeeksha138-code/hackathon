# AI-Based Fake Social Media Account Detection System

![Cybersecurity Shield](https://img.shields.io/badge/Security-Forensic%20AI-blue?style=for-the-badge)
![Python 3.14](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Backend-Flask-black?style=for-the-badge&logo=flask)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?style=for-the-badge&logo=scikitlearn)
![Vanilla JS](https://img.shields.io/badge/Frontend-HTML5%20%2F%20CSS3%20%2F%20JS-yellow?style=for-the-badge&logo=javascript)

A complete, full-stack, machine learning-powered web application built for a **12-Hour College Hackathon**. The system assesses whether a social media account appears **Likely Genuine** or **Suspicious** using observable profile metadata and a trained **Random Forest Classifier**.

---

## 1. Project Overview

Digital platforms are inundated with automated bots, impersonators, and spam farms that manipulate metrics, spread disinformation, and orchestrate fraud.  
The **Fake Account Detector AI** offers a fast, accessible, privacy-first cybersecurity tool where users can evaluate profile authenticity without providing passwords, API tokens, or personal login credentials.

### Key Deliverables:
- **Zero-Credential Forensics:** Evaluates accounts solely using public surface metrics.
- **Probabilistic Risk Scoring:** Generates an empirical 0% to 100% risk percentage derived directly from Scikit-Learn `predict_proba`.
- **Explainable AI Breakdown:** Identifies specific behavioral trust factors and risk flags (e.g., follower asymmetry, username digit entropy, missing avatars).
- **Responsive Dark Cyber Dashboard:** Professional cybersecurity visual styling with animated circular gauges and benchmark comparisons.

---

## 2. Problem Statement & Motivation

Fake social media accounts inflict widespread societal and economic harm:
1. **Disinformation & Astroturfing:** Coordinated bot swarms manipulate public opinion and amplify polarizing content.
2. **Phishing & Financial Fraud:** Impersonation profiles distribute malicious phishing links and cryptocurrency scams.
3. **Advertising Waste & Fraud:** Inauthentic engagement siphons billions of dollars in marketing budgets.
4. **Identity Impersonation:** Cloned accounts compromise individual reputations and harass real users.

---

## 3. Technologies Used

### Frontend
- **HTML5:** Semantic architecture, accessible form controls, and SVG risk meters.
- **CSS3:** Modern dark navy & cyber indigo theme, responsive CSS Grid / Flexbox, glassmorphic cards, and CSS keyframe animations.
- **Vanilla JavaScript (ES6+):** Form validation, asynchronous `fetch()` REST calls, animated risk gauge rendering, and preset profile loaders. *(No heavy frameworks like React/Angular; zero npm build step required).*

### Backend
- **Python 3:** Core programming language.
- **Flask (v3.1):** Lightweight RESTful API server.
- **Flask-CORS:** Enables Cross-Origin Resource Sharing for standalone frontend hosting.

### Machine Learning & Data Processing
- **Scikit-Learn (v1.9):** Random Forest Classifier, model metrics (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`).
- **NumPy:** Vectorized feature array computations.
- **Joblib:** Serialized model persistence (`models/fake_account_model.pkl`).

---

## 4. Folder Structure

```
hackathon/
│
├── frontend/
│   ├── index.html                 # Complete UI: Hero, Form, Results, Architecture, About
│   ├── style.css                  # Cybersecurity styling, gauges, animations, responsiveness
│   └── script.js                  # Fetch API, live validation, gauge animation, presets
│
├── backend/
│   ├── app.py                     # Flask REST API & static file server
│   ├── model.py                   # Inference engine, input validator, explainability logic
│   ├── train_model.py             # ML training pipeline, evaluation, artifact export
│   └── test_app.py                # Automated unit test suite (7 automated tests)
│
├── dataset/
│   ├── generate_dataset.py        # Generates realistic synthetic benchmark dataset
│   ├── synthetic_accounts_dataset.csv # 3,000 balanced benchmark samples
│   └── README.md                  # Dataset documentation & real-world dataset guide
│
├── models/
│   ├── fake_account_model.pkl     # Trained Random Forest classifier pipeline
│   └── model_metrics.json         # Computed test-set evaluation metrics & importances
│
├── requirements.txt               # Python package dependencies
├── run.py                         # One-click startup runner
└── README.md                      # Comprehensive documentation (this file)
```

---

## 5. Machine Learning Features & Algorithm Explanation

### The Simplified User-Friendly Account Features:
1. `followers_count` (int $\ge 0$): Number of followers.
2. `following_count` (int $\ge 0$): Number of followed accounts.
3. `posts_count` (int $\ge 0$): Total public posts/media published.
4. `account_age_bracket` (ordinal $0 \dots 4$): Account age category:
   - `0`: Less than 1 month
   - `1`: 1–6 months
   - `2`: 6–12 months
   - `3`: 1–2 years
   - `4`: More than 2 years
5. `has_bio` (binary: 0 or 1): Does the account have a biography? ($1 = \text{Yes}, 0 = \text{No}$).
6. `has_external_link` (binary: 0 or 1): Does the bio contain an external link? ($1 = \text{Yes}, 0 = \text{No}$).
7. `has_profile_pic` (binary: 0 or 1): Profile picture available? ($1 = \text{Yes}, 0 = \text{No}$).
8. `is_private` (binary: 0 or 1, optional): Account Type ($1 = \text{Private}, 0 = \text{Public}$). Defaults to Public.

### Why Random Forest Classifier?
1. **Non-Linear Interactions:** Bot behaviors depend on non-linear combinations (e.g. following 3,000 accounts with 10 followers is suspicious, but following 3,000 with 100,000 followers is legitimate).
2. **Resistance to Overfitting:** Employs bagging (bootstrap aggregation) over 150 diverse decision trees.
3. **Calibrated Probabilities:** The percentage risk is computed by averaging the tree votes:
   $$\text{Risk Score} = P(Y = \text{Fake} \mid X) \times 100\% = \left( \frac{1}{N_{\text{trees}}} \sum_{t=1}^{N_{\text{trees}}} I(h_t(X) = \text{Fake}) \right) \times 100\%$$
4. **Feature Importances:** Identifies which features exert the greatest predictive influence.

---

## 6. Dataset Information, Transparency & Real-World Mapping

### Transparency Disclosure
The included `dataset/synthetic_accounts_dataset.csv` contains **3,000 mathematically generated samples** modeled after empirical distributions from published social media bot research (e.g., Ferrara et al., Cresci et al.).  
- **Purpose:** Educational prototyping, hackathon demonstration, and architecture verification.
- **Integrity Notice:** Synthetic metrics demonstrate that the end-to-end ML training and inference pipeline operates correctly. They are **not** claimed as universal real-world accuracy against sophisticated nation-state botnets.

### How to Train on Real-World Datasets
If you download an academic or Kaggle dataset (e.g., Kaggle Instagram Fake Spammer Dataset or DARPA/Cresci Bot repository):
1. Place the CSV file into `dataset/` (e.g., `dataset/real_accounts_dataset.csv`).
2. Run the training script:
   ```powershell
   python backend/train_model.py --dataset dataset/real_accounts_dataset.csv
   ```
3. `train_model.py` automatically detects column aliases (e.g., `nb_followers`, `profile_pic`, `is_bot`) and standardizes them into the pipeline.

---

## 7. Windows Setup & Running Instructions

### Prerequisites
- Windows 10 or 11
- Python 3.10+ (Python 3.14 verified)
- VS Code (PowerShell Terminal)

### Step 1: Open the Project in VS Code
Open your terminal in the project root:
```powershell
cd c:\Users\Sudeeksha\Desktop\hackathon
```

### Step 2: Install Dependencies (if not already installed)
```powershell
pip install -r requirements.txt
```

### Step 3: Train the Machine Learning Model
*(Note: A trained model is already bundled in `models/`, but you can re-train anytime)*:
```powershell
python backend/train_model.py
```
*Output will display the test set evaluation results (Accuracy, Precision, Recall, F1, Confusion Matrix) and save `models/fake_account_model.pkl`.*

### Step 4: Run the Automated Test Suite
Verify that all API endpoints, validation routines, and static routes pass:
```powershell
python backend/test_app.py
```

### Step 5: Start the Web Application
You can start the application using either of the following commands:

**Option A (Using the Quick Launcher):**
```powershell
python run.py
```

**Option B (Direct Flask Server):**
```powershell
python backend/app.py
```

### Step 6: Access the Web App
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

> **Tip for Standalone Frontend Development:** If you prefer running the frontend via VS Code Live Server (port 5500), `script.js` has automatic CORS support and connects seamlessly to `http://127.0.0.1:5000`.

---

## 8. API Specification

### `GET /api/health`
- **Description:** Checks backend status and model readiness.
- **Response `200 OK`:**
  ```json
  {
    "status": "healthy",
    "service": "AI-Based Fake Social Media Account Detection System",
    "version": "1.0.0",
    "model_loaded": true,
    "dataset_type": "synthetic_demonstration_dataset",
    "is_demonstration": true,
    "timestamp": "2026-11-06T06:04:14.086069+00:00"
  }
  ```

### `GET /api/metrics`
- **Description:** Returns the evaluated model metrics and feature importance ranking.
- **Response `200 OK`:**
  ```json
  {
    "success": true,
    "metrics": {
      "algorithm": "Random Forest Classifier",
      "accuracy": 1.0,
      "precision": 1.0,
      "recall": 1.0,
      "f1_score": 1.0,
      "feature_importances": [
        {"feature": "posts_count", "importance": 0.4733},
        {"feature": "username_digits_count", "importance": 0.2021},
        {"feature": "account_age_days", "importance": 0.1348}
      ]
    }
  }
  ```

### `POST /api/predict`
- **Description:** Submits account features for inference.
- **Request Body (`application/json`):**
  ```json
  {
    "followers_count": 520,
    "following_count": 410,
    "posts_count": 145,
    "account_age_bracket": 3,
    "has_bio": 1,
    "has_external_link": 0,
    "has_profile_pic": 1,
    "account_type": "public"
  }
  ```
- **Response `200 OK`:**
  ```json
  {
    "success": true,
    "prediction": "Likely Genuine",
    "risk_score": 0.0,
    "risk_level": "Low Risk",
    "status_theme": "success",
    "genuine_probability": 100.0,
    "fake_probability": 0.0,
    "submitted_features": {
      "followers_count": 520,
      "following_count": 410,
      "posts_count": 145,
      "account_age_bracket": 3,
      "account_age_label": "1–2 years",
      "has_bio": 1,
      "has_external_link": 0,
      "has_profile_pic": 1,
      "is_private": 0,
      "account_type_label": "Public"
    },
    "explanation": {
      "summary": "The account features closely align with legitimate, organically grown user profiles with strong trust indicators.",
      "risk_factors": [],
      "trust_factors": [
        {"factor": "Custom Profile Picture", "detail": "Avatar photo uploaded..."},
        {"factor": "Balanced Follower Ratio", "detail": "Follower-to-following ratio is balanced (1.27)..."},
        {"factor": "Established Account (1–2 years)", "detail": "Account tenure is well established..."}
      ]
    }
  }
  ```

---

## 9. Hackathon Demo Walkthrough (For Judges)

When presenting to judges, follow this **3-minute high-impact flow**:

1. **The Hook & Problem (30 sec):**  
   Introduce the project: *"Social media platforms are flooded with automated spam bots, phishing accounts, and fake follower farms. Our system, Fake Account Detector AI, allows anyone to perform fast, privacy-preserving account forensic analysis."*

2. **Zero-Credential Architecture (30 sec):**  
   Highlight that users never disclose passwords or auth tokens. The app evaluates 9 publicly observable metrics.

3. **Live Demonstration via Quick Presets (60 sec):**  
   - Click **"Genuine Everyday User"**: Watch the form auto-fill. Click **"Analyze Account"**. Show the **Green "Likely Genuine"** badge, low risk score, and detected trust signals.
   - Click **"Aggressive Follower Bot"**: Click **"Analyze Account"**. Show the immediate transition to the **Red "Suspicious"** verdict, 95%+ risk score, and detailed risk factors (extreme follower asymmetry, missing avatar, automated digit handle).

4. **Machine Learning & Explainability (45 sec):**  
   Point to the **Architecture Section** and **Behavioral Factors breakdown**. Explain that the score is **not hardcoded**, but computed by a Random Forest ensemble trained on behavioral patterns.

5. **Transparency & Ethics (15 sec):**  
   Point out the educational prototype disclaimer. Emphasize ethical AI by acknowledging that statistical models provide probabilistic estimates rather than definitive identity proof.

---

## 10. Known Limitations & Future Scope

- **Tabular Scope:** Evaluates profile metadata; does not inspect NLP semantics of comments or computer vision image features.
- **Evolving Bot Tactics:** Sophisticated AI bots (e.g. LLM-powered personas) can simulate realistic post histories and bought followings.
- **Future Roadmap:** Integrate graph neural networks (GNNs) for social graph topology analysis and lightweight sentiment analysis on recent posts.

