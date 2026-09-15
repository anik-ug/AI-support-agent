from __future__ import annotations
import subprocess, sys

def run(script):
    print(f"\n===== {script} =====")
    subprocess.run([sys.executable, script], check=True)

def main():
    run("scripts/inspect_dataset.py")
    run("scripts/prepare_data.py")
    run("scripts/generate_taxonomy.py")
    run("scripts/build_index.py")
    print("\nREADY. Next: review data/golden_eval.csv, fill intent/action/difficulty/notes, and ensure data/intent_taxonomy.json contains your reviewed taxonomy.")
    print('Demo: python scripts/run_demo.py "I cannot log in to my account"')

if __name__ == "__main__": main()
