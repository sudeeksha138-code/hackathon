"""
Fake Account Detector AI - Quick Launcher
-----------------------------------------
Executes model readiness checks and launches the Flask web server.
Access the web dashboard directly at: http://127.0.0.1:5000
"""

import os
import sys
import subprocess

def main():
    print("=" * 65)
    print("   FAKE ACCOUNT DETECTOR AI - WEB PLATFORM LAUNCHER")
    print("   College Hackathon Engineering Project")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "fake_account_model.pkl")

    # Step 1: Verify model presence
    if not os.path.exists(model_path):
        print("[!] Model file not detected. Training Random Forest model...")
        train_script = os.path.join(base_dir, "backend", "train_model.py")
        subprocess.run([sys.executable, train_script], check=True)
    else:
        print("[+] Machine learning model verified (models/fake_account_model.pkl).")

    # Step 2: Launch Flask Server
    app_script = os.path.join(base_dir, "backend", "app.py")
    print("[*] Starting Flask Application Server...")
    print("[*] Local URL: http://127.0.0.1:5000")
    print("[*] Press Ctrl+C in this terminal to stop the server.\n")

    subprocess.run([sys.executable, app_script])

if __name__ == "__main__":
    main()

