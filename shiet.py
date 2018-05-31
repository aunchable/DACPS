def mssl(l):
    best = cur = 0
    for i in l:
        cur = max(cur + i, 0)
        best = max(best, cur)
    return best

l = [1, 0, 1, 1]

mssl(l)
