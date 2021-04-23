def compute_bandpass_values(
    rsm,
):
    a = ((1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4), (5, 6), (6, 5))
    b = ((1, 3), (2, 4), (3, 1), (3, 5), (4, 2), (4, 6), (5, 3), (6, 4))
    c = ((1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3))

    a_count = 0
    b_count = 0
    c_count = 0

    for xy in a:
        a_count += rsm[xy]
    a_avr = a_count / len(a)

    for xy in b:
        b_count += rsm[xy]
    b_avr = b_count / len(b)

    for xy in c:
        c_count += rsm[xy]
    c_avr = c_count / len(c)

    return a_avr, b_avr, c_avr
