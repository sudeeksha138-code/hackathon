"""
Dataset Generation Script for Fake Social Media Account Detection
------------------------------------------------------------------
Updated to reflect simplified user-friendly features:
- followers_count: Number of followers (int >= 0)
- following_count: Number of accounts followed (int >= 0)
- posts_count: Number of posts/media uploaded (int >= 0)
- account_age_bracket: Categorical/Ordinal age bracket:
    0: Less than 1 month (< 30 days)
    1: 1–6 months (30–180 days)
    2: 6–12 months (180–365 days)
    3: 1–2 years (365–730 days)
    4: More than 2 years (> 730 days)
- has_bio: Does the account have a biography? (1 = Yes, 0 = No)
- has_external_link: Does the bio contain an external link? (1 = Yes, 0 = No)
- has_profile_pic: Profile picture present (1 = Yes, 0 = No)
- is_private: Account privacy (1 = Private, 0 = Public)
- is_fake: Target Label (0 = Likely Genuine, 1 = Suspicious / Fake)
"""

import os
import csv
import numpy as np

FEATURE_NAMES = [
    "followers_count",
    "following_count",
    "posts_count",
    "account_age_bracket",
    "has_bio",
    "has_external_link",
    "has_profile_pic",
    "is_private"
]
TARGET_NAME = "is_fake"
ALL_COLUMNS = FEATURE_NAMES + [TARGET_NAME]

def generate_synthetic_data(num_samples: int = 3000, random_state: int = 42):
    np.random.seed(random_state)
    n_per_class = num_samples // 2
    
    # -------------------------------------------------------------
    # 1. Genuine Accounts Simulation (Label = 0)
    # -------------------------------------------------------------
    # Genuine users have established accounts, balanced ratios, profile pictures,
    # reasonable post counts, and a realistic mix of public/private accounts.
    
    # Age brackets: mostly older (6-12 mo, 1-2 yr, >2 yr)
    # Probabilities for brackets [0, 1, 2, 3, 4]
    gen_age_bracket = np.random.choice([0, 1, 2, 3, 4], size=n_per_class, p=[0.05, 0.12, 0.20, 0.28, 0.35])
    
    # Following count: typically between 60 and 1500
    gen_following = np.random.lognormal(mean=5.5, sigma=0.8, size=n_per_class).astype(int)
    gen_following = np.clip(gen_following, 10, 2500)
    
    # Followers: reasonable ratio to following (0.5 to 3.0 on average)
    gen_ratio = np.random.lognormal(mean=0.2, sigma=0.7, size=n_per_class)
    gen_followers = (gen_following * gen_ratio).astype(int)
    gen_followers = np.clip(gen_followers, 15, 50000)
    
    # Posts: correlates with age bracket
    base_posts_by_bracket = {0: (1, 8), 1: (5, 30), 2: (15, 80), 3: (30, 200), 4: (60, 600)}
    gen_posts = np.zeros(n_per_class, dtype=int)
    for b in range(5):
        mask = (gen_age_bracket == b)
        count = np.sum(mask)
        if count > 0:
            low, high = base_posts_by_bracket[b]
            gen_posts[mask] = np.random.randint(low, high + 1, size=count)
    
    # Profile pic: ~97% have a profile picture
    gen_has_pic = np.random.choice([1, 0], size=n_per_class, p=[0.97, 0.03])
    
    # Bio: ~90% have a bio
    gen_has_bio = np.random.choice([1, 0], size=n_per_class, p=[0.90, 0.10])
    
    # External link in bio: ~22% of genuine users have a link (portfolio, linktree, etc.)
    # only possible if has_bio == 1
    gen_has_link = np.where(
        gen_has_bio == 1,
        np.random.choice([1, 0], size=n_per_class, p=[0.24, 0.76]),
        0
    )
    
    # Account type: ~35% of genuine everyday personal accounts are Private
    gen_is_private = np.random.choice([1, 0], size=n_per_class, p=[0.36, 0.64])
    
    gen_matrix = np.column_stack([
        gen_followers,
        gen_following,
        gen_posts,
        gen_age_bracket,
        gen_has_bio,
        gen_has_link,
        gen_has_pic,
        gen_is_private,
        np.zeros(n_per_class, dtype=int)
    ])
    
    # -------------------------------------------------------------
    # 2. Suspicious / Fake Accounts Simulation (Label = 1)
    # -------------------------------------------------------------
    # Archetypes:
    # A) Aggressive Follower Spam Bots (young age, huge following, 0-15 followers, no pic)
    # B) Dormant / Scraper Bots (0 posts, 0-5 followers/following, no bio, no pic)
    # C) Phishing / Scam Promo Accounts (young age, external link present, bought followers)
    
    archetypes = np.random.choice(["spam_bot", "scraper_bot", "phishing_bot"], size=n_per_class, p=[0.45, 0.35, 0.20])
    
    fake_followers = np.zeros(n_per_class, dtype=int)
    fake_following = np.zeros(n_per_class, dtype=int)
    fake_posts = np.zeros(n_per_class, dtype=int)
    fake_age_bracket = np.zeros(n_per_class, dtype=int)
    fake_has_bio = np.zeros(n_per_class, dtype=int)
    fake_has_link = np.zeros(n_per_class, dtype=int)
    fake_has_pic = np.zeros(n_per_class, dtype=int)
    fake_is_private = np.zeros(n_per_class, dtype=int)
    
    for i, arch in enumerate(archetypes):
        if arch == "spam_bot":
            # Very young, following thousands, virtually no followers, almost never private
            fake_age_bracket[i] = np.random.choice([0, 1, 2], p=[0.75, 0.20, 0.05])
            fake_following[i] = int(np.random.uniform(900, 4800))
            fake_followers[i] = int(np.random.uniform(0, 35))
            fake_posts[i] = int(np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]))
            fake_has_pic[i] = int(np.random.choice([0, 1], p=[0.65, 0.35]))
            fake_has_bio[i] = int(np.random.choice([0, 1], p=[0.75, 0.25]))
            fake_has_link[i] = int(np.random.choice([0, 1], p=[0.60, 0.40])) if fake_has_bio[i] == 1 else 0
            fake_is_private[i] = 0  # Spam bots need public visibility
            
        elif arch == "scraper_bot":
            # Inactive, 0 posts, no bio, no pic
            fake_age_bracket[i] = np.random.choice([0, 1, 2, 3], p=[0.40, 0.35, 0.15, 0.10])
            fake_following[i] = int(np.random.choice([0, 1, 5, 12, 30], p=[0.3, 0.25, 0.2, 0.15, 0.1]))
            fake_followers[i] = int(np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05]))
            fake_posts[i] = 0
            fake_has_pic[i] = int(np.random.choice([0, 1], p=[0.85, 0.15]))
            fake_has_bio[i] = 0
            fake_has_link[i] = 0
            fake_is_private[i] = int(np.random.choice([0, 1], p=[0.85, 0.15]))
            
        else: # phishing_bot
            # Scam promo: young age, malicious external link in bio, stolen/stock avatar
            fake_age_bracket[i] = np.random.choice([0, 1], p=[0.80, 0.20])
            fake_following[i] = int(np.random.uniform(200, 1400))
            fake_followers[i] = int(np.random.uniform(100, 950))  # bought follower burst
            fake_posts[i] = int(np.random.uniform(1, 6))
            fake_has_pic[i] = 1
            fake_has_bio[i] = 1
            fake_has_link[i] = 1  # Bio link is the phishing payload
            fake_is_private[i] = 0  # Scam pages are public

    # Realistic 6% noise/overlap
    noise_mask = np.random.rand(n_per_class) < 0.06
    for idx in np.where(noise_mask)[0]:
        fake_followers[idx] += int(np.random.uniform(250, 700))
        fake_age_bracket[idx] = int(np.random.choice([2, 3]))
        fake_has_pic[idx] = 1

    fake_matrix = np.column_stack([
        fake_followers,
        fake_following,
        fake_posts,
        fake_age_bracket,
        fake_has_bio,
        fake_has_link,
        fake_has_pic,
        fake_is_private,
        np.ones(n_per_class, dtype=int)
    ])
    
    # Stack and shuffle
    combined = np.vstack([gen_matrix, fake_matrix])
    indices = np.arange(len(combined))
    np.random.shuffle(indices)
    combined = combined[indices]
    
    return combined

def save_to_csv(data: np.ndarray, file_path: str):
    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(ALL_COLUMNS)
        writer.writerows(data)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "synthetic_accounts_dataset.csv")
    
    print("[*] Generating updated simplified synthetic demonstration dataset (3,000 samples)...")
    dataset_matrix = generate_synthetic_data(num_samples=3000, random_state=42)
    save_to_csv(dataset_matrix, output_path)
    
    genuine_count = int(np.sum(dataset_matrix[:, -1] == 0))
    fake_count = int(np.sum(dataset_matrix[:, -1] == 1))
    
    print(f"[+] Dataset saved to: {output_path}")
    print(f"[*] Total rows: {len(dataset_matrix)}")
    print(f"[*] Class distribution: Genuine = {genuine_count}, Suspicious = {fake_count}")
    print(f"[*] Features: {FEATURE_NAMES}")
    print(f"[*] Sample first row: {dataset_matrix[0]}")
