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
