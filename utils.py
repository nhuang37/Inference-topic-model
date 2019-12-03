import torch
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix

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

def create_per_class_tables(loader, model, device, class_names, threshold=0.5):
    """
    Function that creates per class tables with count, TN, FN, TP, FP, precision, recall, f1.
    @param: loader - data loader for the dataset to test against
    """
    model.eval()
    outputs_list_nc = []
    true_list_nc = []
    with torch.no_grad():
        for data, length, labels in loader:
            data_batch, length_batch, label_batch = data.to(device), length.to(device), labels.float()
            outputs_bc = torch.sigmoid(model(data_batch, length_batch))
            outputs_bc = outputs_bc.detach().cpu().numpy().astype(np.float)
            outputs_bc = (outputs_bc > threshold)
            outputs_list_nc.append(outputs_bc)
            true_list_nc.append(label_batch.detach().cpu().numpy().astype(np.float))
    # to np.array
    outputs_list_nc = np.vstack(outputs_list_nc)
    true_list_nc = np.vstack(true_list_nc)
    
    # per class counts
    counts_c = true_list_nc.sum(axis=0)
    
    # per class confusion matrix: TN, FN, TP, FP
    confusion_matrix_c22 = multilabel_confusion_matrix(
        true_list_nc,
        outputs_list_nc,
    )
    confusion_matrix_c4 = confusion_matrix_c22.reshape(-1, 4)
    
    # per class precision, recall, f-score
    precision_c, recall_c, f1_c, _ = precision_recall_fscore_support(
        true_list_nc,
        outputs_list_nc,
        average=None
    )

    # combine all metrics in a dict
    per_class_metrics = {
        "class_name": class_names,
        "count": counts_c,
        "TN": confusion_matrix_c4[:, 0],
        "FN": confusion_matrix_c4[:, 2],
        "TP": confusion_matrix_c4[:, 3],
        "FP": confusion_matrix_c4[:, 1],
        "precision": precision_c,
        "recall": recall_c,
        "f1": f1_c
    }
    return pd.DataFrame(per_class_metrics)