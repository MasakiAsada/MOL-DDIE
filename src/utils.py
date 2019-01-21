def calculate_microF(pred, gold, negative_indx):
    numer, precision_denom, recall_denom = 0, 0, 0

    for p, g in zip(pred, gold):
        if p != negative_indx:
            precision_denom += 1
            if p == g:
                numer += 1
        if g != negative_indx:
            recall_denom += 1

    precision = numer / (precision_denom + 0.00001)
    recall = numer / (recall_denom + 0.00001)
    microF = 2 * precision * recall / (precision + recall + 0.00001)

    return precision, recall, microF

