"""다중 모델(LGBM/XGBoost/CatBoost) 단일 폴드 적합기. subject_id 는 범주형으로 처리.

각 함수는 검증/테스트의 클래스 1 확률을 반환한다. 작은 데이터(폴드당 ~360행)에 맞춘 강한 정규화.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

CAT = "subject_id"

LGB_PARAMS = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=25,
                  feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l1=1.0, lambda_l2=1.0, verbosity=-1)
XGB_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", learning_rate=0.02,
                  max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.5,
                  reg_alpha=1.0, reg_lambda=2.0, tree_method="hist", enable_categorical=True)
CAT_PARAMS = dict(loss_function="Logloss", learning_rate=0.03, depth=4, l2_leaf_reg=6.0,
                  random_strength=1.0, bootstrap_type="Bernoulli", subsample=0.8,
                  verbose=False, allow_writing_files=False)


def _lgb(Xtr, ytr, Xva, yva, Xte, seed):
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=[CAT], free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=[CAT], free_raw_data=False)
    m = lgb.train({**LGB_PARAMS, "seed": seed}, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
    return m.predict(Xva), m.predict(Xte)


def _xgb(Xtr, ytr, Xva, yva, Xte, seed):
    m = xgb.XGBClassifier(n_estimators=3000, early_stopping_rounds=80, random_state=seed,
                          **XGB_PARAMS)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]


def _cat(Xtr, ytr, Xva, yva, Xte, seed):
    Xtr = Xtr.copy(); Xva = Xva.copy(); Xte = Xte.copy()
    for d in (Xtr, Xva, Xte):
        d[CAT] = d[CAT].astype(str)
    m = CatBoostClassifier(iterations=3000, early_stopping_rounds=80, random_seed=seed,
                           cat_features=[CAT], **CAT_PARAMS)
    m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]


FITTERS = {"lgb": _lgb, "xgb": _xgb, "cat": _cat}


def fit_fold(model_type, Xtr, ytr, Xva, yva, Xte, seed=42):
    return FITTERS[model_type](Xtr, ytr, Xva, yva, Xte, seed)
