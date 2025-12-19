import warnings
warnings.filterwarnings("ignore")  # 屏蔽不太重要的 warning

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# ======================
# 1. 读入数据（改成你的路径）
# ======================
train_path = "/root/autodl-tmp/Regional-Prompting-FLUX/task1_ds_004/train.csv"
test_path  = "/root/autodl-tmp/Regional-Prompting-FLUX/task1_ds_004/test.csv"

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

# 列名统一成字符串，避免 int/str 不一致
train.columns = [str(c) for c in train.columns]
test.columns  = [str(c) for c in test.columns]

print("Train columns:", train.columns.tolist())
print("Test  columns:", test.columns.tolist())

if "label" not in train.columns:
    raise ValueError("train.csv 中找不到 label 列，请确认列名（区分大小写）")


# ======================
# 2. 拆分特征 / 标签
# ======================
feature_cols = [c for c in train.columns if c != "label"]
feature_cols = sorted(feature_cols)  # 固定顺序

X_raw = train[feature_cols]
y = train["label"]
X_test_raw = test[feature_cols]

num_classes = y.nunique()
print("特征数:", len(feature_cols))
print("Train X_raw shape:", X_raw.shape)
print("y 分布:\n", y.value_counts().sort_index())


# ======================
# 3. 特征工程：异常值裁剪 + 标准化
# ======================
def fit_preprocessor(X_train_raw):
    """
    在训练特征上拟合：异常值边界 + 标准化 scaler
    """
    # 1) 计算每一列的 1% 和 99% 分位数，后面用来裁剪异常值
    lower = X_train_raw.quantile(0.01)
    upper = X_train_raw.quantile(0.99)

    # 2) 先裁剪，再在裁剪后的数据上拟合 StandardScaler
    X_train_clipped = X_train_raw.clip(lower=lower, upper=upper, axis=1)

    scaler = StandardScaler()
    scaler.fit(X_train_clipped)

    return lower, upper, scaler


def transform_preprocess(X_raw, lower, upper, scaler):
    """
    使用已经拟合好的边界和 scaler，对任意一份数据（train/val/test）做同样变换
    返回：
      - X_clipped: 只做了异常值裁剪（给树模型用）
      - X_scaled:  剪裁后再做标准化（给 MLP 用）
    """
    X_clipped = X_raw.clip(lower=lower, upper=upper, axis=1)
    X_scaled = scaler.transform(X_clipped)
    return X_clipped, X_scaled


# ======================
# 4. 先划分一个验证集，用来评估各模型和集成的表现
# ======================
X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
    X_raw, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 在训练子集上拟合特征工程
lower_tr, upper_tr, scaler_tr = fit_preprocessor(X_tr_raw)

# 对 train / val 做同样的特征处理
X_tr_clipped, X_tr_scaled = transform_preprocess(X_tr_raw, lower_tr, upper_tr, scaler_tr)
X_val_clipped, X_val_scaled = transform_preprocess(X_val_raw, lower_tr, upper_tr, scaler_tr)

print("完成特征工程：异常值裁剪 + 标准化（训练/验证）。")


# ======================
# 5. 定义三个基模型：XGBoost / RandomForest / MLP
# ======================

# 5.1 XGBoost（主力模型）
xgb_params = dict(
    n_estimators=800,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    reg_lambda=2.0,
    reg_alpha=0.0,
    gamma=0.0,
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    tree_method="hist",   # 以后有 GPU 可以改 "gpu_hist"
    random_state=42
)
xgb_clf = XGBClassifier(**xgb_params)

# 5.2 随机森林（和 XGB 结构不同，增加多样性）
rf_clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)

# 5.3 MLP 神经网络（用标准化后的特征）
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    batch_size=256,
    learning_rate_init=1e-3,
    max_iter=300,
    early_stopping=True,      # 自动早停
    n_iter_no_change=10,
    validation_fraction=0.1,
    random_state=42
)


# ======================
# 6. 在验证集上分别训练并评估三个模型
# ======================
print("\n===== 在验证集上训练并评估基模型 =====")

# 6.1 XGBoost
xgb_clf.fit(X_tr_clipped, y_tr)
val_pred_xgb = xgb_clf.predict(X_val_clipped)
acc_xgb = accuracy_score(y_val, val_pred_xgb)
print(f"XGBoost Validation Accuracy: {acc_xgb:.4f}")

# 6.2 RandomForest
rf_clf.fit(X_tr_clipped, y_tr)
val_pred_rf = rf_clf.predict(X_val_clipped)
acc_rf = accuracy_score(y_val, val_pred_rf)
print(f"RandomForest Validation Accuracy: {acc_rf:.4f}")

# 6.3 MLP（用标准化后的特征）
mlp_clf.fit(X_tr_scaled, y_tr)
val_pred_mlp = mlp_clf.predict(X_val_scaled)
acc_mlp = accuracy_score(y_val, val_pred_mlp)
print(f"MLP Validation Accuracy: {acc_mlp:.4f}")


# ======================
# 7. 集成学习：用概率加权平均做集成预测
# ======================
print("\n===== 集成（加权概率平均）评估 =====")

# 三个模型在验证集上的类别概率
proba_xgb = xgb_clf.predict_proba(X_val_clipped)
proba_rf  = rf_clf.predict_proba(X_val_clipped)
proba_mlp = mlp_clf.predict_proba(X_val_scaled)

# 给 XGBoost 更高权重，RF/MLP 次之
w_xgb = 2.0
w_rf  = 1.0
w_mlp = 1.0

proba_ensemble = (w_xgb * proba_xgb + w_rf * proba_rf + w_mlp * proba_mlp) / (w_xgb + w_rf + w_mlp)
val_pred_ens = proba_ensemble.argmax(axis=1)

acc_ens = accuracy_score(y_val, val_pred_ens)
print(f"Ensemble Validation Accuracy: {acc_ens:.4f}")


# ======================
# 8. （可选）5 折交叉验证估计整体平均水平
# ======================
print("\n===== 5 折交叉验证估计集成的平均表现（会比单次 val 更稳定） =====")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_ens = []

for fold, (idx_tr, idx_va) in enumerate(cv.split(X_raw, y), 1):
    X_tr_f_raw, X_va_f_raw = X_raw.iloc[idx_tr], X_raw.iloc[idx_va]
    y_tr_f, y_va_f = y.iloc[idx_tr], y.iloc[idx_va]

    # 每折重新拟合特征工程
    lower_f, upper_f, scaler_f = fit_preprocessor(X_tr_f_raw)
    X_tr_f_clipped, X_tr_f_scaled = transform_preprocess(X_tr_f_raw, lower_f, upper_f, scaler_f)
    X_va_f_clipped, X_va_f_scaled = transform_preprocess(X_va_f_raw, lower_f, upper_f, scaler_f)

    # 三个基模型
    xgb_f = XGBClassifier(**xgb_params)
    rf_f  = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    mlp_f = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=42
    )

    # 训练
    xgb_f.fit(X_tr_f_clipped, y_tr_f)
    rf_f.fit(X_tr_f_clipped, y_tr_f)
    mlp_f.fit(X_tr_f_scaled, y_tr_f)

    # 概率集成
    p_xgb_f = xgb_f.predict_proba(X_va_f_clipped)
    p_rf_f  = rf_f.predict_proba(X_va_f_clipped)
    p_mlp_f = mlp_f.predict_proba(X_va_f_scaled)

    p_ens_f = (w_xgb * p_xgb_f + w_rf * p_rf_f + w_mlp * p_mlp_f) / (w_xgb + w_rf + w_mlp)
    y_va_pred_f = p_ens_f.argmax(axis=1)

    acc_f = accuracy_score(y_va_f, y_va_pred_f)
    cv_scores_ens.append(acc_f)
    print(f"Fold {fold} Ensemble Accuracy: {acc_f:.4f}")

cv_mean_ens = sum(cv_scores_ens) / len(cv_scores_ens)
print("5 折 CV Ensemble mean Accuracy:", cv_mean_ens)


# ======================
# 9. 最终模型：用全部训练集 + 特征工程训练三模型，然后在 test 上做集成预测
# ======================
print("\n===== 用全部训练集 + 特征工程 训练最终集成模型，并预测 test =====")

# 在全量特征上拟合特征工程
lower_full, upper_full, scaler_full = fit_preprocessor(X_raw)
X_full_clipped, X_full_scaled = transform_preprocess(X_raw, lower_full, upper_full, scaler_full)
X_test_clipped, X_test_scaled = transform_preprocess(X_test_raw, lower_full, upper_full, scaler_full)

# 全量训练三个模型
xgb_final = XGBClassifier(**xgb_params)
rf_final  = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
mlp_final = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    batch_size=256,
    learning_rate_init=1e-3,
    max_iter=300,
    early_stopping=True,
    n_iter_no_change=10,
    validation_fraction=0.1,
    random_state=42
)

xgb_final.fit(X_full_clipped, y)
rf_final.fit(X_full_clipped, y)
mlp_final.fit(X_full_scaled, y)

# 三模型在 test 上的概率
p_test_xgb = xgb_final.predict_proba(X_test_clipped)
p_test_rf  = rf_final.predict_proba(X_test_clipped)
p_test_mlp = mlp_final.predict_proba(X_test_scaled)

p_test_ens = (w_xgb * p_test_xgb + w_rf * p_test_rf + w_mlp * p_test_mlp) / (w_xgb + w_rf + w_mlp)
test_pred = p_test_ens.argmax(axis=1)

# 覆盖 test 里的 label 列，保留所有特征列
test_with_pred = test.copy()
test_with_pred["label"] = test_pred

out_path = "test_with_pred_ensemble.csv"
test_with_pred.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"预测完成，已保存到 {out_path}")
