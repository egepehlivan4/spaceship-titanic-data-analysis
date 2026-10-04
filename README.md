# Spaceship Titanic — Feature Representation vs. Model Complexity

Code accompanying the paper *"Feature Representation Outweighs Model Complexity
in Tabular Classification: An Empirical Study on the Spaceship Titanic
Dataset."*

The study compares nine classification algorithms, one Optuna-tuned
configuration and two ensembling strategies on identical cross-validation
folds, tests the differences with a paired Wilcoxon signed-rank test, and
ablates each engineered feature block under the same protocol.

## Headline result

| | Accuracy | ROC-AUC |
|---|---|---|
| Best model (tuned CatBoost) | 0.8172 ± 0.0065 | 0.9082 |
| Out-of-fold (tuned CatBoost) | 0.8165 | 0.9083 |

Neither hyperparameter optimisation (p = 0.770), nor stacking (p = 0.793), nor
soft voting (p = 0.313) improved on a single well-configured gradient boosting
model. Removing the engineered spending features cost 6.4 accuracy points
(p = 0.002).

## Repository layout

```
.
├── 01_data_preparation.ipynb        # EDA, imputation, feature engineering
├── 02_modeling_and_evaluation.ipynb # models, tuning, ensembles, tests, ablation
├── data/
│   ├── train.csv                    # from Kaggle (not tracked)
│   ├── test.csv                     # from Kaggle (not tracked)
│   └── X_train.csv, X_test.csv, y_train.csv, test_ids.csv   # generated
├── submission.csv                   # generated
├── requirements.txt
└── README.md
```

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download `train.csv` and `test.csv` from the
[competition page](https://www.kaggle.com/competitions/spaceship-titanic)
into `data/`, then run the notebooks in order:

```bash
jupyter nbconvert --to notebook --execute 01_data_preparation.ipynb --inplace
jupyter nbconvert --to notebook --execute 02_modeling_and_evaluation.ipynb --inplace
```

Notebook 1 writes the feature matrices to `data/`; notebook 2 reads them and
writes `submission.csv`. Total runtime is roughly four minutes on a single CPU
core.

## Evaluation protocol

All models are scored by stratified 5-fold cross-validation repeated over two
random seeds (10 evaluations per model). Fold assignments are identical across
models, so scores are paired and compared with a Wilcoxon signed-rank test.

Hyperparameter search uses a **separate inner 3-fold split with its own seed**;
the selected configuration is re-scored on the outer protocol, which the search
never observed. The best inner-search score is never reported as a performance
estimate.

## Leakage boundary

Training and test records are concatenated before feature construction so that
group- and family-level lookups see the full passenger manifest. Only predictor
columns are shared — the target is separated before concatenation and is used
in no imputation, encoding or aggregation step. This transductive treatment of
the covariates is standard for a competition whose test predictors are
published, and it is stated explicitly rather than left implicit.

## Authors

Ege Pehlivan, Turan Mamedovi — Department of Computer Engineering, Faculty of
Technology, Gazi University.

## License

MIT
