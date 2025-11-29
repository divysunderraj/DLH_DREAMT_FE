# load packages
import pandas as pd
import numpy as np
import random
import shap


from utils import *
from models import *
from datasets import *

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Prepare the data
# Adjust your path here
quality_df_dir = './results/quality_scores_per_subject.csv'
features_dir = "dataset_sample/features_df/"
info_dir = "dataset_sample/participant_info.csv"
clean_df, new_features, good_quality_sids = data_preparation(
    threshold = 0.2, 
    quality_df_dir = quality_df_dir,
    features_dir = features_dir,
    info_dir = info_dir)

# Split data to train, validation, and test set
SW_df, final_features = split_data(clean_df, good_quality_sids, new_features)

random.seed(0)
train_sids = random.sample(good_quality_sids, 56)
remaining_sids = [subj for subj in good_quality_sids if subj not in train_sids]
val_sids = random.sample(remaining_sids, 8)
test_sids = [subj for subj in remaining_sids if subj not in val_sids]

group_variables = ["AHI_Severity", "Obesity"]
# when idx == 0, it returns ['AHI_Severity'], the first variable in the list
# when idx == 1, it returns ['Obesity'], the second variable in the list
group_variable = get_variable(group_variables, idx=0)

X_train, y_train, group_train = train_test_split(SW_df, train_sids, final_features, group_variable)
X_val, y_val, group_val = train_test_split(SW_df, val_sids, final_features, group_variable)
X_test, y_test, group_test = train_test_split(SW_df, test_sids, final_features, group_variable)

# Resample all the data
X_train_resampled, y_train_resampled, group_train_resampled = resample_data(X_train, y_train, group_train, group_variable)

final_lgb_model = LightGBM_engine(X_train_resampled, y_train_resampled, X_val, y_val)

# ---- LightGBM Train Probabilities ----
lgb_prob_ls_train, lgb_len_train, lgb_true_ls_train = compute_probabilities(
    train_sids, SW_df, final_features, "lgb", final_lgb_model, group_variable
)
lgb_train_results_df = LightGBM_result(
    final_lgb_model, X_train, y_train, lgb_prob_ls_train, lgb_true_ls_train
)

# ---- LightGBM Test Probabilities ----
lgb_prob_ls_test, lgb_len_test, lgb_true_ls_test = compute_probabilities(
    test_sids, SW_df, final_features, "lgb", final_lgb_model, group_variable
)
lgb_test_results_df = LightGBM_result(
    final_lgb_model, X_test, y_test, lgb_prob_ls_test, lgb_true_ls_test
)

# ---- SHAP for LightGBM ----
explainer = shap.TreeExplainer(final_lgb_model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names=final_features)


# =============================
# LSTM ON TOP OF LIGHTGBM
# =============================
lgb_dataloader_train = LSTM_dataloader(
    lgb_prob_ls_train, lgb_len_train, lgb_true_ls_train, batch_size=32
)
lgb_dataloader_test = LSTM_dataloader(
    lgb_prob_ls_test, lgb_len_test, lgb_true_ls_test, batch_size=1
)

lgb_LSTM_model = LSTM_engine(
    lgb_dataloader_train, num_epoch=5, hidden_layer_size=32, learning_rate=0.001
)
lgb_lstm_test_results_df = LSTM_eval(
    lgb_LSTM_model, lgb_dataloader_test, lgb_true_ls_test, "LightGBM_LSTM"
)


# =============================
# GPBOOST MODEL
# =============================
final_gpb_model = GPBoost_engine(
    X_train_resampled, group_train_resampled, y_train_resampled,
    X_val, y_val, group_val
)

# ---- GPBoost Train Probabilities ----
gpb_prob_ls_train, gpb_len_train, gpb_true_ls_train = compute_probabilities(
    train_sids, SW_df, final_features, 'gpb', final_gpb_model, group_variable
)
gpb_train_results_df = GPBoost_result(
    final_gpb_model, X_train, y_train, group_train,
    gpb_prob_ls_train, gpb_true_ls_train
)

# ---- GPBoost Test Probabilities ----
gpb_prob_ls_test, gpb_len_test, gpb_true_ls_test = compute_probabilities(
    test_sids, SW_df, final_features, 'gpb', final_gpb_model, group_variable
)
gpb_test_results_df = GPBoost_result(
    final_gpb_model, X_test, y_test, group_test,
    gpb_prob_ls_test, gpb_true_ls_test
)


# =============================
# LSTM ON TOP OF GPBOOST
# =============================
gpb_dataloader_train = LSTM_dataloader(
    gpb_prob_ls_train, gpb_len_train, gpb_true_ls_train, batch_size=32
)
gpb_dataloader_test = LSTM_dataloader(
    gpb_prob_ls_test, gpb_len_test, gpb_true_ls_test, batch_size=1
)

gpb_LSTM_model = LSTM_engine(
    gpb_dataloader_train, num_epoch=5, hidden_layer_size=32, learning_rate=0.001
)
gpb_lstm_test_results_df = LSTM_eval(
    gpb_LSTM_model, gpb_dataloader_test, gpb_true_ls_test, "GPBoost_LSTM"
)


# =============================
# TRANSFORMER ON TOP OF LIGHTGBM
# =============================
lgb_transformer_train = LSTM_dataloader(
    lgb_prob_ls_train, lgb_len_train, lgb_true_ls_train, batch_size=32
)
lgb_transformer_test = LSTM_dataloader(
    lgb_prob_ls_test, lgb_len_test, lgb_true_ls_test, batch_size=1
)

Transformer_model_lgb = Transformer_engine(
    lgb_transformer_train,
    num_epoch=5,
    d_model=128,
    nhead=8,
    hidden_dim=256,
    learning_rate=0.001
)

lgb_transformer_test_results_df = Transformer_eval(
    Transformer_model_lgb,
    lgb_transformer_test,
    lgb_true_ls_test,
    "LightGBM_Transformer"
)


# =============================
# TRANSFORMER ON TOP OF GPBOOST
# =============================
gpb_transformer_train = LSTM_dataloader(
    gpb_prob_ls_train, gpb_len_train, gpb_true_ls_train, batch_size=32
)
gpb_transformer_test = LSTM_dataloader(
    gpb_prob_ls_test, gpb_len_test, gpb_true_ls_test, batch_size=1
)

Transformer_model_gpb = Transformer_engine(
    gpb_transformer_train,
    num_epoch=5,
    d_model=128,
    nhead=8,
    hidden_dim=256,
    learning_rate=0.001
)

gpb_transformer_test_results_df = Transformer_eval(
    Transformer_model_gpb,
    gpb_transformer_test,
    gpb_true_ls_test,
    "GPBoost_Transformer"
)


# =============================
# COMBINE ALL RESULTS
# =============================
overall_result = pd.concat([
    lgb_test_results_df,
    lgb_lstm_test_results_df,
    lgb_transformer_test_results_df,
    gpb_test_results_df,
    gpb_lstm_test_results_df,
    gpb_transformer_test_results_df
])

overall_result.to_csv("overall_result.csv")
print(group_variable)
print(overall_result)