from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def evaluate(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return {
        'ROC-AUC':   round(roc_auc_score(y, proba), 3),
        'Precision': round(precision_score(y, pred), 3),
        'Recall':    round(recall_score(y, pred), 3),
        'F1':        round(f1_score(y, pred), 3),
    }