def calculate_accuracy(pred, gold):
    n = len(pred)
    true_cnt = 0
    for p, g in zip(pred, gold):
        if p == g:
            true_cnt += 1
    return true_cnt / n

