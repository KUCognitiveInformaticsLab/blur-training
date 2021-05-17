def compute_bandpass_values(
    rsm,
):
    r1 = ((1, 2), (2, 3), (3, 4), (4, 5), (5, 6))
    l1 = ((2, 1), (3, 2), (4, 3), (5, 4), (6, 5))
    r2 = ((1, 3), (2, 4), (3, 5), (4, 6))
    l2 = ((3, 1), (4, 2), (5, 3), (6, 4))
    r3 = ((1, 4), (2, 5), (3, 6))
    l3 = ((4, 1), (5, 2), (6, 3))
    r4 = ((1, 5), (2, 6))
    l4 = ((5, 1), (6, 2))
    r5 = [(1, 6)]
    l5 = [(6, 1)]

    r1_count = 0
    r2_count = 0
    r3_count = 0
    r4_count = 0
    r5_count = 0

    l1_count = 0
    l2_count = 0
    l3_count = 0
    l4_count = 0
    l5_count = 0

    for xy in r1:
        r1_count += rsm[xy]
    r1_avr = r1_count / len(r1)

    for xy in l1:
        l1_count += rsm[xy]
    l1_avr = l1_count / len(l1)

    for xy in r2:
        r2_count += rsm[xy]
    r2_avr = r2_count / len(r2)

    for xy in l2:
        l2_count += rsm[xy]
    l2_avr = l2_count / len(l2)

    for xy in r3:
        r3_count += rsm[xy]
    r3_avr = r3_count / len(r3)

    for xy in l3:
        l3_count += rsm[xy]
    l3_avr = l3_count / len(l3)

    for xy in r4:
        r4_count += rsm[xy]
    r4_avr = r4_count / len(r4)

    for xy in l4:
        l4_count += rsm[xy]
    l4_avr = l4_count / len(l4)

    for xy in r5:
        r5_count += rsm[xy]
    r5_avr = r5_count / len(r5)

    for xy in l5:
        l5_count += rsm[xy]
    l5_avr = l5_count / len(l5)

    avr0 = 1
    avr1 = ((r1_avr * len(r1)) + (l1_avr * len(l1))) / (len(r1) + len(l1))
    avr2 = ((r2_avr * len(r2)) + (l2_avr * len(l2))) / (len(r2) + len(l2))
    avr3 = ((r3_avr * len(r3)) + (l3_avr * len(l3))) / (len(r3) + len(l3))
    avr4 = ((r4_avr * len(r4)) + (l4_avr * len(l4))) / (len(r4) + len(l4))
    avr5 = ((r5_avr * len(r5)) + (l5_avr * len(l5))) / (len(r5) + len(l5))

    return [l3_avr, l2_avr, l1_avr, avr0, r1_avr, r2_avr, r3_avr]
    # return [avr0, avr1, avr2, avr3, avr4, avr5]


def compute_bandpass_values_0(
    rsm,
):
    a = ((1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4), (5, 6), (6, 5))
    b = ((1, 3), (2, 4), (3, 1), (3, 5), (4, 2), (4, 6), (5, 3), (6, 4))
    c = ((1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3))

    r1_count = 0
    b_count = 0
    c_count = 0

    for xy in a:
        r1_count += rsm[xy]
    r1_avr = r1_count / len(a)

    for xy in b:
        b_count += rsm[xy]
    b_avr = b_count / len(b)

    for xy in c:
        c_count += rsm[xy]
    c_avr = c_count / len(c)

    return r1_avr, b_avr, c_avr


def compute_bandpass_x(y):
    y0, y1, y2, y3 = y[3:]
    y4 = 0
    if y1 < 0.5 <= y0:
        return 1 / ((y0 - y1) * 2)
    elif y2 < 0.5 <= y1:
        return 1 / ((y1 - y2) * 2)
    elif y3 < 0.5 <= y2:
        return 1 / ((y2 - y3) * 2)
    elif y4 < 0.5 <= y3:
        return 1 / ((y3 - y4) * 2)
