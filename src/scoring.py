"""Полный пайплайн скоринга: сырые транзакции -> сегмент + вероятность оттока.

Загружает обученные артефакты (этапы 3 и 5) и применяет их к новым данным,
ничего не переобучая. Переиспользует compute_features из features.py.
"""
import os
import joblib
import pandas as pd

from features import compute_features

MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')

CLUSTER_NAMES = {0: 'Новые', 1: 'Лояльные', 2: 'Обычные', 3: 'Группа риска'}


def _load(name):
    return joblib.load(os.path.join(MODELS, name))


def score_customers(transactions):
    """transactions: DataFrame с колонками Invoice, Quantity, InvoiceDate,
    Price, Customer ID. Возвращает (scored, X_scaled, model, meta), где
    scored — таблица на клиента с Cluster_name, churn_proba, churn_label.
    """
    df = transactions.copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # 1. Очистка (как этап 2)
    df = df.dropna(subset=['Customer ID'])
    df = df[df['Price'] > 0]

    # 2. Опорная дата = "снимок" загруженных данных
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # 3. Признаки (RFM + поведенческие) — та же функция, что в пайплайне
    feats = compute_features(df, reference_date)

    # 4. Фильтр выбросов теми же границами (этап 3)
    bounds = _load('outlier_bounds.pkl')
    n_before = len(feats)
    feats = feats[
        (feats['Monetary'] < bounds['upper_monetary']) &
        (feats['Frequency'] < bounds['upper_frequency'])
    ].copy()
    n_outliers = n_before - len(feats)

    # 5. Кластер: scaler + kmeans с этапа 3 (transform/predict, не переобучаем)
    clu_scaler = _load('scaler.pkl')
    kmeans = _load('kmeans_model.pkl')
    feats['Cluster'] = kmeans.predict(
        clu_scaler.transform(feats[['Recency', 'Frequency', 'Monetary']])
    )
    feats['Cluster_name'] = feats['Cluster'].map(CLUSTER_NAMES)

    # 6. Churn: строим матрицу признаков как при обучении (этап 5)
    meta = _load('churn_meta.pkl')
    model = _load('churn_model.pkl')
    churn_scaler = _load('churn_scaler.pkl')

    X = pd.get_dummies(
        feats.drop(columns=['Cluster_name']), columns=['Cluster'], drop_first=True
    ).astype(float)
    # выравниваем колонки под обученную модель (на случай отсутствующих кластеров)
    X = X.reindex(columns=meta['features'], fill_value=0.0)
    X[meta['num_cols']] = churn_scaler.transform(X[meta['num_cols']])

    proba = model.predict_proba(X)[:, 1]
    feats['churn_proba'] = proba.round(3)
    feats['churn_label'] = (proba >= meta['threshold']).astype(int)

    feats.attrs['n_outliers'] = n_outliers
    feats.attrs['reference_date'] = reference_date
    feats.attrs['threshold'] = meta['threshold']
    return feats, X, model, meta
