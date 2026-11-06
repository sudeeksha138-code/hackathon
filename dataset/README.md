# Dataset Documentation

## Overview & Transparency Notice

> **Important Disclosure:**  
> This project currently ships with a **synthetic demonstration dataset** (`synthetic_accounts_dataset.csv`) generated mathematically using behavioral distributions reported in peer-reviewed social media bot detection literature (e.g., Ferrara et al., Cresci et al.).  
>  
> While the statistical properties reflect realistic patterns (such as follower-to-following ratios, account age brackets, profile completeness, external link presence, and privacy settings), **this synthetic dataset is intended strictly for prototyping, system integration testing, and hackathon presentation**. We do **not** claim that metrics obtained from this synthetic data reflect universal real-world performance against evolving botnets.

---

## Dataset Schema (Updated Simplified Schema)

The dataset contains **3,000 samples** with a balanced 50:50 distribution (1,500 Likely Genuine, 1,500 Suspicious/Bot).

| Column Name | Data Type | Value Range | Description |
| :--- | :--- | :--- | :--- |
| `followers_count` | Integer | $0 \dots 50,000+$ | Total number of accounts following this profile. |
| `following_count` | Integer | $0 \dots 5,000+$ | Total number of accounts this profile follows. |
| `posts_count` | Integer | $0 \dots 15,000+$ | Number of public posts/media published by this account. |
| `account_age_bracket` | Ordinal Integer | $0 \dots 4$ | Account age category:<br>• `0`: Less than 1 month<br>• `1`: 1–6 months<br>• `2`: 6–12 months<br>• `3`: 1–2 years<br>• `4`: More than 2 years |
| `has_bio` | Binary (0/1) | $0$ or $1$ | Whether the user has entered any biography ($1 = \text{Yes}, 0 = \text{No}$). |
| `has_external_link` | Binary (0/1) | $0$ or $1$ | Whether the bio contains an external link ($1 = \text{Yes}, 0 = \text{No}$). |
| `has_profile_pic` | Binary (0/1) | $0$ or $1$ | Whether an avatar/profile image is uploaded ($1 = \text{Yes}, 0 = \text{No}$). |
| `is_private` | Binary (0/1) | $0$ or $1$ | Whether the account is Private ($1 = \text{Private}, 0 = \text{Public}$). |
| `is_fake` | Binary (0/1) | $0$ or $1$ | Target classification label ($0 = \text{Likely Genuine}, 1 = \text{Suspicious/Fake}$). |

---

## Where to Obtain Real-World Public Datasets

If you wish to benchmark or train this system on real social media accounts:
1. **Kaggle Instagram Fake Spammer Dataset**: [Kaggle: Instagram Fake and Automated Account Detection](https://www.kaggle.com/datasets/freehackers/instagram-fake-spammer-genuine-accounts)
2. **Cresci et al. Bot Repository (MIB Datasets)**: [Botometer / Cresci Datasets](https://botometer.osome.iu.edu/bot-repository/datasets.html)
3. **Kaggle Social Media Fake Accounts Classification**: [Kaggle Social Network Fake User Accounts](https://www.kaggle.com/datasets)

---

## How to Plug in a Real Dataset

To train the model on your own or downloaded real-world CSV:
```powershell
python backend/train_model.py --dataset dataset/real_accounts_dataset.csv
```
The automated preprocessing pipeline in `backend/train_model.py` maps common column aliases into the standardized 8 features.
