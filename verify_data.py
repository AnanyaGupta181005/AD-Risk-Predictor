import os
from pathlib import Path

# Path to your data
BASE_DIR = Path("data/raw")

def check_folder(name, path):
    if path.exists() and any(path.iterdir()):
        print(f"✅ {name}: FOUND")
        return True
    else:
        print(f"❌ {name}: MISSING or EMPTY (Path: {path})")
        return False

print("--- DATASET CHECK ---")
d_ok = check_folder("DRIVE", BASE_DIR / "DRIVE")
s_ok = check_folder("STARE", BASE_DIR / "STARE")
c_ok = check_folder("CHASE", BASE_DIR / "CHASE")
f_ok = check_folder("Framingham", BASE_DIR / "framingham")

if d_ok and s_ok and c_ok and f_ok:
    print("\nSUCCESS: All data is ready. You can now run Phase 2.")
else:
    print("\nSTILL MISSING DATA: Please download and place the folders as shown above.")