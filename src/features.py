import pandas as pd

def compute_rfm(data, reference_date):
    data = data.copy()
    data['Total sum'] = data['Price'] * data['Quantity']
    data['Days ago'] = (reference_date - data['InvoiceDate']).dt.days
    rfm = data.groupby('Customer ID').agg(
    Recency = ('Days ago', 'min'),
    Frequency = ('Invoice', 'nunique'),
    Monetary = ('Total sum','sum')
    )
    rfm = rfm[rfm['Monetary'] > 0]
    return rfm

def compute_features(data, reference_date):
    # 1. RFM-база: Recency, Frequency, Monetary на клиента (индекс = Customer ID)
    features = compute_rfm(data, reference_date)

    # 2. Таблица заказов: одна строка на (клиент, Invoice) с датой заказа.
    #    reset_index() возвращает Customer ID и Invoice из индекса в колонки.
    order_dates = (
        data.groupby(['Customer ID', 'Invoice'])['InvoiceDate']
        .min()
        .reset_index()
    )

    # 3. Tenure — сколько дней клиент "с нами" на момент reference_date.
    #    Берём первую (минимальную) дату заказа каждого клиента и вычитаем из опорной.
    first_purchase = order_dates.groupby('Customer ID')['InvoiceDate'].min()
    tenure = (reference_date - first_purchase).dt.days

    # 4. PurchaseRate — темп покупок: сколько заказов на день "стажа".
    #    clip(lower=1) страхует от деления на 0 (клиент, чья первая покупка в день T).
    features['Tenure'] = tenure
    features['PurchaseRate'] = features['Frequency'] / features['Tenure'].clip(lower=1)

    # 5. Ритм покупок: интервалы между соседними заказами клиента.
    #    Сортируем по клиенту и дате (diff требует хронологического порядка),
    #    считаем разницу с предыдущим заказом ВНУТРИ клиента (groupby не даёт
    #    вычитать через границу клиентов), берём средний интервал на клиента.
    order_dates = order_dates.sort_values(['Customer ID', 'InvoiceDate'])
    order_dates['interval'] = (
        order_dates.groupby('Customer ID')['InvoiceDate'].diff().dt.days
    )
    interval_mean = order_dates.groupby('Customer ID')['interval'].mean()

    features['IntervalMean'] = interval_mean
    # Одноразовые клиенты: интервала нет (NaN). Заполняем их Tenure — оценкой
    # "интервала наблюдения" (мы видели их лишь раз за столько дней).
    features['IntervalMean'] = features['IntervalMean'].fillna(features['Tenure'])

    # Флаг одноразовика — чтобы модель отличала заполненные значения от реальных.
    features['SinglePurchase'] = (features['Frequency'] == 1).astype(int)

    # 6. Recency относительно СВОЕГО ритма: клиент "просрочен" по своим меркам?
    #    >1 — молчит дольше обычного (риск), <1 — в пределах привычного.
    features['RecencyRatio'] = features['Recency'] / features['IntervalMean'].clip(lower=1)

    return features
