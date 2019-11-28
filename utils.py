from sklearn.metrics import precision_recall_fscore_support

def get_metrics_dict(y_true, y_pred):
    """
    Help function that tests the model's performance on a dataset.
    """
    # macro precision, recall, f-score
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro"
    )
    # micro precision, recall, f-score
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro"
    )
    # combine all metrics in a dict
    dict_metrics = {
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro, 
        "recall_micro": recall_micro, 
        "f1_micro": f1_micro,
    }
    # round
    n_digits = 3
    dict_metrics = {
        metric_name: round(value, n_digits) 
        for metric_name, value in dict_metrics.items()
    }
    return dict_metrics