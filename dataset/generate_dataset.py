"""
Dataset Generation Script for Fake Social Media Account Detection
------------------------------------------------------------------
Generates a realistic, statistically grounded benchmark dataset with
both core surface parameters and engineered forensic features:
  1. followers_count (int >= 0)
  2. following_count (int >= 0)
  3. posts_count (int >= 0)
  4. account_age_bracket (0: <1 mo, 1: 1-6 mo, 2: 6-12 mo, 3: 1-2 yr, 4: >2 yr)
  5. has_bio (1 = Yes, 0 = No)
  6. has_external_link (1 = Yes, 0 = No)
  7. has_profile_pic (1 = Yes, 0 = No)
  8. is_private (1 = Private, 0 = Public)
Engineered Forensic Features:
  9. follower_following_ratio = followers / (following + 1)
  10. follow_asymmetry = (following - followers) / (following + followers + 1)
  11. post_rate_per_age = posts / (account_age_bracket + 1)
Target:
  12. is_fake (0 = Genuine, 1 = Suspicious / Fake)
"""

import os
import csv
import numpy as np

RAW_FEATURE_NAMES = [
    "followers_count",
    "following_count",
    "posts_count",
    "account_age_bracket",
    "has_bio",
    "has_external_link",
    "has_profile_pic",
    "is_private"
]

ENGINEERED_FEATURE_NAMES = [
    "follower_following_ratio",
    "follow_asymmetry",
    "post_rate_per_age"
]

FEATURE_NAMES = RAW_FEATURE_NAMES + ENGINEERED_FEATURE_NAMES
TARGET_NAME = "is_fake"
ALL_COLUMNS = FEATURE_NAMES + [TARGET_NAME]


def compute_engineered_features(
    followers: np.ndarray,
    following: np.ndarray,
    posts: np.ndarray,
    age_bracket: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes non-linear behavioral ratios for tabular classification."""
    ratio = followers / (following + 1.0)
    asymmetry = (following - followers) / (following + followers + 1.0)
    post_rate = posts / (age_bracket + 1.0)
    return ratio, asymmetry, post_rate


def generate_synthetic_data(num_samples: int = 4000, random_state: int = 42) -> np.ndarray:
    np.random.seed(random_state)
    n_per_class = num_samples // 2

    # =========================================================================
    # 1. Genuine Accounts (Label = 0)
    # =========================================================================
    # Archetypes:
    # A) Everyday Personal Users (65%): Balanced follow ratio, moderate activity
    # B) Brand New Genuine Users (15%): Low tenure, modest posts (0-15), balanced ratio
    # C) Creators & Public Figures (20%): High followers, established tenure, public
    
    gen_arch = np.random.choice(["personal", "new_user", "creator"], size=n_per_class, p=[0.65, 0.15, 0.20])
    
    gen_followers = np.zeros(n_per_class, dtype=int)
    gen_following = np.zeros(n_per_class, dtype=int)
    gen_posts = np.zeros(n_per_class, dtype=int)
    gen_age = np.zeros(n_per_class, dtype=int)
    gen_bio = np.zeros(n_per_class, dtype=int)
    gen_link = np.zeros(n_per_class, dtype=int)
    gen_pic = np.zeros(n_per_class, dtype=int)
    gen_private = np.zeros(n_per_class, dtype=int)

    for i, arch in enumerate(gen_arch):
        if arch == "personal":
            gen_age[i] = int(np.random.choice([1, 2, 3, 4], p=[0.10, 0.25, 0.35, 0.30]))
            gen_following[i] = int(np.clip(np.random.lognormal(mean=5.4, sigma=0.6), 40, 1800))
            # Ratio 0.4 to 2.2
            r = np.random.uniform(0.4, 2.2)
            gen_followers[i] = int(np.clip(gen_following[i] * r, 20, 5000))
            # Posts correlate with age
            max_p = {1: 45, 2: 120, 3: 280, 4: 600}[gen_age[i]]
            min_p = {1: 2, 2: 10, 3: 20, 4: 40}[gen_age[i]]
            gen_posts[i] = int(np.random.randint(min_p, max_p + 1))
            gen_pic[i] = int(np.random.choice([1, 0], p=[0.98, 0.02]))
            gen_bio[i] = int(np.random.choice([1, 0], p=[0.92, 0.08]))
            gen_link[i] = int(np.random.choice([1, 0], p=[0.18, 0.82]))
            gen_private[i] = int(np.random.choice([1, 0], p=[0.48, 0.52]))

        elif arch == "new_user":
            # Just joined (< 1 month or 1-6 months)
            gen_age[i] = int(np.random.choice([0, 1], p=[0.60, 0.40]))
            gen_following[i] = int(np.random.randint(15, 160))
            gen_followers[i] = int(np.random.randint(8, 140))
            # Genuine new users post a modest initial amount, NEVER 100+ posts in 2 weeks!
            gen_posts[i] = int(np.random.randint(0, 16))
            gen_pic[i] = int(np.random.choice([1, 0], p=[0.94, 0.06]))
            gen_bio[i] = int(np.random.choice([1, 0], p=[0.85, 0.15]))
            gen_link[i] = int(np.random.choice([1, 0], p=[0.08, 0.92]))
            gen_private[i] = int(np.random.choice([1, 0], p=[0.55, 0.45]))

        else: # creator
            gen_age[i] = int(np.random.choice([2, 3, 4], p=[0.20, 0.40, 0.40]))
            gen_following[i] = int(np.random.randint(150, 1500))
            # Ratio 2.5 to 40.0
            r = np.random.uniform(2.5, 35.0)
            gen_followers[i] = int(np.clip(gen_following[i] * r, 1200, 85000))
            gen_posts[i] = int(np.random.randint(60, 1400))
            gen_pic[i] = 1
            gen_bio[i] = 1
            gen_link[i] = int(np.random.choice([1, 0], p=[0.75, 0.25]))
            gen_private[i] = 0  # creators are public

    gen_ratio, gen_asym, gen_rate = compute_engineered_features(
        gen_followers, gen_following, gen_posts, gen_age
    )

    gen_matrix = np.column_stack([
        gen_followers,
        gen_following,
        gen_posts,
        gen_age,
        gen_bio,
        gen_link,
        gen_pic,
        gen_private,
        gen_ratio,
        gen_asym,
        gen_rate,
        np.zeros(n_per_class, dtype=int)
    ])

    # =========================================================================
    # 2. Suspicious / Fake Accounts (Label = 1)
    # =========================================================================
    # Archetypes:
    # A) Mass-Follower Bots (35%): Follows 800-6000+, followers 0-45 (ratio < 0.04),
    #    can have 0-300 posts, default avatar, public or private!
    # B) Content Spammer / Repost Bots (25%): High posts (50-400) on YOUNG accounts,
    #    severe follower asymmetry, mass automated uploads
    # C) Dormant / Scraper / Zombie Bots (20%): 0 posts, 0-5 followers, 0-30 following,
    #    no avatar, no bio, public or private
    # D) Phishing & Affiliate Scam Accounts (20%): Young tenure, external link present,
    #    bought or zero followers, aggressive outreach
    
    fake_arch = np.random.choice(["mass_follower", "content_spammer", "dormant", "phishing"], size=n_per_class, p=[0.35, 0.25, 0.20, 0.20])

    fake_followers = np.zeros(n_per_class, dtype=int)
    fake_following = np.zeros(n_per_class, dtype=int)
    fake_posts = np.zeros(n_per_class, dtype=int)
    fake_age = np.zeros(n_per_class, dtype=int)
    fake_bio = np.zeros(n_per_class, dtype=int)
    fake_link = np.zeros(n_per_class, dtype=int)
    fake_pic = np.zeros(n_per_class, dtype=int)
    fake_private = np.zeros(n_per_class, dtype=int)

    for i, arch in enumerate(fake_arch):
        if arch == "mass_follower":
            # Severe follower asymmetry: follows hundreds or thousands, virtually no followers
            fake_age[i] = int(np.random.choice([0, 1, 2], p=[0.70, 0.22, 0.08]))
            fake_following[i] = int(np.random.randint(600, 5500))
            fake_followers[i] = int(np.random.randint(0, 35))
            # Some mass follower bots post spam media (0 to 250 posts)
            fake_posts[i] = int(np.random.choice([0, 1, 2, 5, 24, 85, 140, 220], p=[0.35, 0.20, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03]))
            fake_pic[i] = int(np.random.choice([0, 1], p=[0.70, 0.30]))
            fake_bio[i] = int(np.random.choice([0, 1], p=[0.75, 0.25]))
            fake_link[i] = int(np.random.choice([0, 1], p=[0.65, 0.35]))
            # Bots can be public or private! (many bots are private follow-traps)
            fake_private[i] = int(np.random.choice([0, 1], p=[0.72, 0.28]))

        elif arch == "content_spammer":
            # Automated bot that reposts scraped media rapidly on brand new account
            fake_age[i] = int(np.random.choice([0, 1], p=[0.80, 0.20]))
            fake_following[i] = int(np.random.randint(800, 4500))
            fake_followers[i] = int(np.random.randint(2, 60))
            fake_posts[i] = int(np.random.randint(45, 350))  # 45 to 350 posts in < 1 month!
            fake_pic[i] = int(np.random.choice([0, 1], p=[0.60, 0.40]))
            fake_bio[i] = int(np.random.choice([0, 1], p=[0.70, 0.30]))
            fake_link[i] = int(np.random.choice([0, 1], p=[0.50, 0.50]))
            fake_private[i] = int(np.random.choice([0, 1], p=[0.75, 0.25]))

        elif arch == "dormant":
            # Zombie / scraper account
            fake_age[i] = int(np.random.choice([0, 1, 2, 3], p=[0.35, 0.35, 0.20, 0.10]))
            fake_following[i] = int(np.random.choice([0, 1, 3, 10, 35], p=[0.3, 0.3, 0.2, 0.1, 0.1]))
            fake_followers[i] = int(np.random.choice([0, 1, 3], p=[0.7, 0.2, 0.1]))
            fake_posts[i] = 0
            fake_pic[i] = int(np.random.choice([0, 1], p=[0.85, 0.15]))
            fake_bio[i] = 0
            fake_link[i] = 0
            fake_private[i] = int(np.random.choice([0, 1], p=[0.80, 0.20]))

        else: # phishing
            # Young account with external scam link in bio
            fake_age[i] = int(np.random.choice([0, 1], p=[0.85, 0.15]))
            fake_following[i] = int(np.random.randint(200, 2200))
            fake_followers[i] = int(np.random.choice([np.random.randint(5, 50), np.random.randint(200, 800)], p=[0.7, 0.3]))
            fake_posts[i] = int(np.random.randint(1, 40))
            fake_pic[i] = int(np.random.choice([1, 0], p=[0.75, 0.25]))
            fake_bio[i] = int(np.random.choice([1, 0], p=[0.85, 0.15]))
            fake_link[i] = 1  # Bio link is the phishing link
            fake_private[i] = int(np.random.choice([0, 1], p=[0.85, 0.15]))

    # Realistic overlap/noise (~5%)
    noise_mask = np.random.rand(n_per_class) < 0.05
    for idx in np.where(noise_mask)[0]:
        fake_followers[idx] += int(np.random.uniform(200, 600))
        fake_age[idx] = int(np.random.choice([2, 3]))
        fake_pic[idx] = 1

    fake_ratio, fake_asym, fake_rate = compute_engineered_features(
        fake_followers, fake_following, fake_posts, fake_age
    )

    fake_matrix = np.column_stack([
        fake_followers,
        fake_following,
        fake_posts,
        fake_age,
        fake_bio,
        fake_link,
        fake_pic,
        fake_private,
        fake_ratio,
        fake_asym,
        fake_rate,
        np.ones(n_per_class, dtype=int)
    ])

    combined = np.vstack([gen_matrix, fake_matrix])
    indices = np.arange(len(combined))
    np.random.shuffle(indices)
    return combined[indices]


def save_to_csv(data: np.ndarray, file_path: str):
    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(ALL_COLUMNS)
        for row in data:
            formatted_row = [
                int(row[0]),  # followers_count
                int(row[1]),  # following_count
                int(row[2]),  # posts_count
                int(row[3]),  # account_age_bracket
                int(row[4]),  # has_bio
                int(row[5]),  # has_external_link
                int(row[6]),  # has_profile_pic
                int(row[7]),  # is_private
                round(float(row[8]), 5),   # follower_following_ratio
                round(float(row[9]), 5),   # follow_asymmetry
                round(float(row[10]), 5),  # post_rate_per_age
                int(row[11])  # is_fake
            ]
            writer.writerow(formatted_row)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "synthetic_accounts_dataset.csv")

    print("[*] Generating upgraded forensic demonstration dataset (4,000 samples)...")
    dataset_matrix = generate_synthetic_data(num_samples=4000, random_state=42)
    save_to_csv(dataset_matrix, output_path)

    genuine_count = int(np.sum(dataset_matrix[:, -1] == 0))
    fake_count = int(np.sum(dataset_matrix[:, -1] == 1))

    print(f"[+] Dataset saved to: {output_path}")
    print(f"[*] Total rows: {len(dataset_matrix)}")
    print(f"[*] Class distribution: Genuine = {genuine_count}, Suspicious = {fake_count}")
    print(f"[*] Features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")
