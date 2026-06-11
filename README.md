# Stretching the Manifold

**The Ambient-Intrinsic Dimension Gap as a Universal Predictor of Machine Learning Algorithm Behavior**

This repository contains the code, experiments, and paper for a study on how the ratio between ambient and intrinsic dimensionality (ρ = d_a / d_i) predicts the failure or success of machine learning algorithms in high-dimensional settings.

## Abstract

High-dimensional data is routinely analyzed using algorithms designed for low-dimensional settings, yet predicting when such algorithms will fail remains an open problem. Prior work has studied failure as a function of ambient dimension or intrinsic dimension separately, but neither alone provides a reliable predictor.

We propose the **stretching ratio** ρ = d_a / d_i — the ratio of ambient to intrinsic dimension — as a unified predictor of algorithm behavior. Through controlled experiments across 45 conditions, we show that ρ follows a power law relationship with k-means clustering failure, with a universal phase transition at ρ* ≈ 3.5–5 independent of intrinsic dimension.

Strikingly, ρ has the **opposite** effect on distance-weighted local methods: kNN classification and regression *improve* monotonically with ρ, revealing a geometric tradeoff rather than a simple curse.

## Key Findings

1. **Power law decay**: k-means clustering performance (ARI) decays as ARI(ρ) ≈ α·ρ^(-β)
2. **Universal failure threshold**: ρ* ≈ 3.5–5 across all tested intrinsic dimensions
3. **Logarithmic scaling law**: β(d_i) = 0.140·ln(d_i) + 0.534 (prediction error < 0.025)
4. **Algorithm-dependent reversal**: kNN classification and regression *improve* with ρ (opposite of k-means)
5. **Outperforms existing complexity measures**: log(ρ) achieves R²=0.556, vs. 0.321 for participation ratio, 0.313 for explained variance ratio, and 0.147 for ambient/intrinsic dimension alone
6. **Heavy-tailed noise shifts the threshold**: under t-distributed noise (df=2), ρ* drops below 2

## Repository Structure

```
stretching-the-manifold/
├── paper/
│   ├── sn-article.tex          # Full paper (SN Computer Science format)
│   └── figures/                # All figures used in the paper
├── src/
│   ├── data_generator.py       # Synthetic dataset generation + TwoNN estimator
│   ├── experiment.py           # Main k-means scaling experiment
│   ├── experiment_knn.py       # kNN classification/regression experiment
│   ├── experiment_heavytail.py # Heavy-tailed noise experiment
│   ├── experiment_baselines.py # Baseline complexity measure comparisons
│   └── plot_paper.py           # Generates all paper figures
├── results/
│   ├── results.json
│   ├── results_knn.json
│   ├── results_heavytail.json
│   └── results_baselines.json
└── requirements.txt
```

## Reproducing the Experiments

```bash
pip install -r requirements.txt

# Main k-means scaling experiment
python src/experiment.py

# kNN classification and regression
python src/experiment_knn.py

# Heavy-tailed noise analysis
python src/experiment_heavytail.py

# Baseline complexity measure comparison
python src/experiment_baselines.py

# Generate all figures
python src/plot_paper.py
```

## Paper

The full paper is available in [`paper/sn-article.tex`](paper/sn-article.tex), submitted to SN Computer Science (Springer) and posted on arXiv.

## Citation

```bibtex
@article{amanov2026stretching,
  title={Stretching the Manifold: The Ambient-Intrinsic Dimension Gap as a Universal Predictor of Machine Learning Algorithm Behavior},
  author={Amanov, Dayanch},
  year={2026},
  note={Under review}
}
```

## License

MIT
