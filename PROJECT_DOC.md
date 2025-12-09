# Project Overview

- transformer_model.py: PyTorch Transformer model for churn prediction, plus training/inference helpers.
- feature_pipeline.py: Reusable feature engineering pipeline that builds event-level tensors and encodes categories.
- xgboost_model/xgb_train.py: Standalone XGBoost training/prediction script with a simple CLI entrypoint.
- xgboost_model/xgb_user_features.py: Aggregates event rows into user-level features for the XGBoost baseline.
- feature_engineering.ipynb: Notebook capturing exploratory feature work that mirrors the reusable pipeline.
- model_construction.ipynb: Notebook experiments around model choices, loss functions, and training behavior.
- EDA_test.ipynb / Ds_project.ipynb: Additional exploratory data analysis notebooks kept for reference.
- data_features.md: Notes on target definition and high-level dataset observations.
- churn-prediction-25-26/: Directory containing the raw train/test parquet files used for experiments.
- transformer_best.pt: Saved Transformer checkpoint; training_loss.png: plot of training loss; submission.csv: sample prediction output.
