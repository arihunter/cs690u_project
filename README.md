# Predicting Phylogenetic Distances

A minimal implementation of alignment‑free regression models to predict phylogenetic distances between archaeal RpoB nucleotide sequences using 3‑mer frequencies and basic sequence statistics.

## Setup & Run

```bash
# 1. Create and activate a virtual environment
python -m venv cs690_env
source cs690_env/bin/activate      # Windows PowerShell: cs690_env\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Execute the analysis
python main.py
