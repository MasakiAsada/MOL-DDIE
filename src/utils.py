def count(y, t, r_d, p_d, p_n):
    for i in range(len(t)):
        if t[i] != 0:
            r_d += 1.0
        if y[i] != 0:
            p_d += 1.0
            if y[i] == t[i]:
                p_n += 1.0
    return r_d, p_d, p_n

def calculate_microF(r_d, p_d, p_n):
    P, R = p_n / (p_d + 0.00001), p_n / (r_d + 0.00001)
    F = 2 * P * R / (P + R + 0.00001)
    return P*100, R*100, F*100

def count_true(y, t):
    pos = 0
    '''
    for i in range(len(t)):
        if y[i] == t[i]:
            pos += 1
    return pos
    '''
    for i in range(len(t)):
        if y[i] >= 0:
            if 1 == t[i]:
                pos += 1
        else:
            if -1 == t[i]:
                pos += 1
    return pos
