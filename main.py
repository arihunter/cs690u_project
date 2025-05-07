#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import numpy as np
from datasets import load_dataset
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# 1. Load the two tables
dist_ds = load_dataset("tattabio/rpob_arch_dna_phylogeny_distances", split="train")
seq_ds  = load_dataset("tattabio/rpob_arch_dna_phylogeny_sequences", split="train")

# 2. Build lookup from accession to sequence
seq_map = { rec["Entry"]: rec["Sequence"] for rec in seq_ds }

# 3. Prepare paired examples
pairs = []
for rec in dist_ds:
    s1 = seq_map[rec["ID1"]]
    s2 = seq_map[rec["ID2"]]
    pairs.append((s1, s2, float(rec["distance"])))
df_pairs = pd.DataFrame(pairs, columns=["seq1","seq2","distance"])

# 4. Feature functions
def kmer_vector(seq, k=3):
    """Normalized k-mer frequency vector."""
    kmers = [''.join(p) for p in product('ACGT', repeat=k)]
    counts = dict.fromkeys(kmers, 0)
    total = len(seq) - k + 1
    for i in range(total):
        sub = seq[i:i+k]
        if sub in counts:
            counts[sub] += 1
    return np.array([counts[kmer]/total for kmer in kmers]), kmers

def simple_feats(seq):
    """Length, GC, AT, Shannon entropy."""
    L = len(seq)
    c = {b: seq.count(b) for b in 'ACGT'}
    gc = (c['G'] + c['C']) / L
    at = (c['A'] + c['T']) / L
    freqs = np.array([c[b] / L for b in 'ACGT'])
    H = -np.sum([p * np.log2(p) for p in freqs if p > 0])
    return np.array([L, gc, at, H]), ["length", "gc_content", "at_content", "entropy"]

# 5. Build feature matrix and names
all_features = []
feature_names = []

# First get k-mer names
_, kmer_names = kmer_vector("AAA"*10)  # dummy seq to get kmer_names

# Then build actual data
for s1, s2, _ in pairs:
    v1, _ = kmer_vector(s1)
    v2, _ = kmer_vector(s2)
    f1, simple_names = simple_feats(s1)
    f2, _ = simple_feats(s2)

    # compute abs diffs
    kdiff = np.abs(v1 - v2)
    sdiff = np.abs(f1 - f2)

    # collect names for first instance
    if not feature_names:
        # prefix k-mer diffs
        feature_names.extend([f"{kmer}_diff" for kmer in kmer_names])
        # prefix simple-feature diffs
        feature_names.extend([f"{name}_diff" for name in simple_names])

    all_features.append(np.concatenate([kdiff, sdiff]))

X = np.vstack(all_features)
y = df_pairs["distance"].values

# 6. Split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train & evaluate
models = {
    "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=5),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42),
    "Linear Regression": LinearRegression()
}

for name, model in models.items():
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    r, _ = pearsonr(y_te, preds)
    print(f"{name}: Pearson r = {r:.4f}")

    if hasattr(model, "feature_importances_"):
        imps = pd.Series(model.feature_importances_, index=feature_names)
        top = imps.nlargest(15)
        print(f"\nTop 15 importances for {name}:")
        print(top.to_string())
        top.plot.barh(title=f"{name} – Top 15 Feature Importances")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    elif hasattr(model, "coef_"):
        coefs = pd.Series(model.coef_, index=feature_names)
        top = coefs.abs().nlargest(15).index
        print(f"\nTop 15 coefficients for {name}:")
        print(coefs.loc[top].to_string())
        coefs.loc[top].plot.barh(title=f"{name} – Top 15 Coefficients")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
