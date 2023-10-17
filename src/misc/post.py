def calculate_metrics(y_true, y_pred):
    y_true_and_pred = (y_true & y_pred).sum()
    y_pred_sum = y_pred.sum()
    y_true_sum = y_true.sum()
    return {
        'accuracy' :  ((y_true == y_pred).sum() / y_true.numel()).item(),
        'precision' : ((y_true_and_pred).sum() / y_pred_sum).item() if y_pred_sum > 0 else 0,
        'recall' :    ((y_true_and_pred).sum() / y_true_sum).item() if y_true_sum > 0 else 0,
    }