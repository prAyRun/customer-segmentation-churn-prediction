"""Streamlit-демо: сегментация клиентов + предсказание оттока.

Запуск:  streamlit run app/app.py
"""
import io
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, '..', 'src'))
from scoring import score_customers  # noqa: E402

EXAMPLE_PATH = os.path.join(BASE, 'sample_transactions.csv')

st.set_page_config(page_title='Отток и сегменты', layout='wide')
st.title('🛒 Сегментация клиентов и предсказание оттока')
st.caption(
    'Загрузите CSV с транзакциями (Invoice, Quantity, InvoiceDate, Price, Customer ID) — '
    'получите сегмент, вероятность оттока и факторы риска по каждому клиенту.'
)


@st.cache_data(show_spinner='Считаю скоринг и SHAP...')
def score_and_explain(source_key, file_bytes=None):
    if source_key == 'example':
        tx = pd.read_csv(EXAMPLE_PATH)
    else:
        tx = pd.read_csv(io.BytesIO(file_bytes))
    scored, X, model, meta = score_customers(tx)
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer(X)
    return scored, X, shap_values, meta['threshold']


# --- источник данных ---
st.sidebar.header('Данные')
uploaded = st.sidebar.file_uploader('CSV с транзакциями', type=['csv'])
use_example = st.sidebar.checkbox('Демо на примере (Online Retail II)', value=uploaded is None)

if uploaded is not None and not use_example:
    scored, X, shap_values, threshold = score_and_explain('upload', uploaded.getvalue())
elif use_example:
    scored, X, shap_values, threshold = score_and_explain('example')
else:
    st.info('⬅️ Загрузите CSV или включите демо-пример в сайдбаре.')
    st.stop()

# --- сводка ---
n = len(scored)
at_risk = int(scored['churn_label'].sum())
c1, c2, c3 = st.columns(3)
c1.metric('Клиентов оценено', f'{n:,}')
c2.metric('В зоне риска оттока', f'{at_risk:,}', f'{100 * at_risk / n:.0f}%')
c3.metric('Порог отсечки', f'{threshold:.2f}')

st.divider()

# --- распределение по кластерам ---
left, right = st.columns([1, 1])
with left:
    st.subheader('Сегменты клиентов')
    counts = scored['Cluster_name'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(counts.index, counts.values, color='#4C78A8')
    ax.set_ylabel('клиентов')
    plt.xticks(rotation=20, ha='right')
    st.pyplot(fig, clear_figure=True)
with right:
    st.subheader('Отток по сегментам')
    seg = scored.groupby('Cluster_name')['churn_label'].mean().mul(100).round(1)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(seg.index, seg.values, color='#E45756')
    ax.set_ylabel('% в риске')
    plt.xticks(rotation=20, ha='right')
    st.pyplot(fig, clear_figure=True)
    st.caption('Доля клиентов в зоне риска внутри каждого сегмента, %')

st.divider()

# --- таблица рисков ---
st.subheader('Клиенты по риску оттока')
table = (
    scored[['Cluster_name', 'Recency', 'Frequency', 'Monetary', 'churn_proba', 'churn_label']]
    .sort_values('churn_proba', ascending=False)
    .rename(columns={'churn_proba': 'P(отток)', 'churn_label': 'В риске'})
)
st.dataframe(
    table,
    use_container_width=True,
    height=320,
    column_config={
        'P(отток)': st.column_config.ProgressColumn(
            'P(отток)', min_value=0.0, max_value=1.0, format='%.2f'
        )
    },
)

st.divider()

# --- разбор конкретного клиента (SHAP) ---
st.subheader('Почему клиент в зоне риска? (SHAP)')
ids = list(table.index)
selected = st.selectbox('Клиент (по умолчанию — самый рискованный):', ids)

i = X.index.get_loc(selected)
row = scored.loc[selected]

m1, m2, m3, m4 = st.columns(4)
m1.metric('Сегмент', row['Cluster_name'])
m2.metric('P(отток)', f"{row['churn_proba']:.2f}")
m3.metric('Recency (дней)', int(row['Recency']))
m4.metric('Frequency', int(row['Frequency']))

shap.plots.waterfall(shap_values[i], show=False)
st.pyplot(plt.gcf(), clear_figure=True)
st.caption(
    'Каждый признак толкает риск вверх (к оттоку) или вниз относительно среднего по выборке. '
    'Значения признаков показаны масштабированными — важно направление вклада.'
)
