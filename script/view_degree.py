# Plot the distribution of degrees
# Set D1 - D4 by hand.

import matplotlib.pyplot as plt

D1 = [(1, 26), (2, 23), (3, 27), (4, 25), (5, 27), (6, 28), (7, 27), (8, 25), (9, 23), (10, 22), (11, 23), (12, 20), (13, 29), (14, 31), (15, 23), (16, 25), (17, 31), (18, 27), (19, 24), (20, 26), (21, 30), (22, 24), (23, 22), (24, 28), (25, 24), (26, 26), (27, 25), (28, 29), (29, 23), (30, 26), (31, 21), (32, 23), (33, 26), (34, 28), (35, 28), (36, 23), (37, 22), (38, 21), (39, 25), (40, 26), (41, 26), (42, 27), (43, 28), (44, 28), (45, 26), (46, 24), (47, 24), (48, 21), (49, 26), (50, 27), (51, 20), (52, 28), (53, 17), (54, 25), (55, 22), (56, 27), (57, 25), (58, 24), (59, 28), (60, 28), (61, 19)]
D2 = [(1, 11), (2, 16), (3, 14), (4, 15), (5, 12), (6, 16), (7, 14), (8, 12), (9, 12), (10, 11), (11, 13), (12, 16), (13, 19), (14, 20), (15, 13), (16, 13), (17, 14), (18, 14), (19, 14), (20, 13), (21, 18), (22, 9), (23, 21), (24, 12), (25, 19), (26, 15), (27, 11), (28, 16), (29, 16), (30, 17), (31, 12), (32, 14), (33, 14), (34, 15), (35, 12), (36, 14), (37, 13), (38, 16), (39, 12), (40, 16), (41, 19), (42, 20), (43, 15), (44, 14), (45, 12), (46, 13), (47, 15), (48, 12), (49, 21), (50, 23), (51, 17), (52, 18), (53, 10), (54, 15), (55, 20), (56, 15), (57, 15), (58, 16), (59, 14), (60, 16), (61, 10)]
D3 = [(1, 19), (2, 21), (3, 21), (4, 20), (5, 26), (6, 27), (7, 28), (8, 22), (9, 21), (10, 19), (11, 29), (12, 20), (13, 32), (14, 28), (15, 20), (16, 21), (17, 28), (18, 26), (19, 23), (20, 23), (21, 27), (22, 19), (23, 28), (24, 30), (25, 29), (26, 22), (27, 19), (28, 25), (29, 27), (30, 32), (31, 26), (32, 25), (33, 23), (34, 26), (35, 23), (36, 26), (37, 31), (38, 28), (39, 28), (40, 18), (41, 31), (42, 21), (43, 23), (44, 29), (45, 20), (46, 22), (47, 18), (48, 23), (49, 25), (50, 20), (51, 25), (52, 27), (53, 18), (54, 24), (55, 28), (56, 27), (57, 25), (58, 23), (59, 23), (60, 26), (61, 20)]
D4 = [(1, 22), (2, 19), (3, 18), (4, 26), (5, 23), (6, 27), (7, 28), (8, 24), (9, 22), (10, 25), (11, 27), (12, 25), (13, 25), (14, 19), (15, 22), (16, 28), (17, 17), (18, 23), (19, 29), (20, 22), (21, 22), (22, 14), (23, 30), (24, 25), (25, 25), (26, 26), (27, 21), (28, 26), (29, 38), (30, 31), (31, 22), (32, 20), (33, 26), (34, 30), (35, 26), (36, 19), (37, 26), (38, 21), (39, 26), (40, 20), (41, 24), (42, 25), (43, 27), (44, 23), (45, 30), (46, 19), (47, 26), (48, 24), (49, 24), (50, 28), (51, 30), (52, 20), (53, 31), (54, 16), (55, 24), (56, 21), (57, 27), (58, 26), (59, 16), (60, 23), (61, 25)]


plt.xlim(0,60)
plt.hist([v[1] for v in D1],label="baseline")
plt.hist([v[1] for v in D2],label="mild")
plt.hist([v[1] for v in D3],label="moderate")
plt.hist([v[1] for v in D4],label="recovery")

plt.legend()
plt.savefig("deg_29.png")