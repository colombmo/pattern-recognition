import numpy as np
from scipy.spatial.distance import euclidean

# from dtw import dtw
from fastdtw import fastdtw
import datetime
import os

# Read feature vectors from .txt
# Fill dictionary of couples id - feature vectors
features = {}
enrollment = {}
previous = {}

# Get name of signatures files
with open("users.txt", "r") as myfile:
    lines = myfile.readlines()

for fn in lines:
    filenum = fn.replace("\n", "")
    for i in range(1, 6):
        with open("enrollment/" + filenum + "-g-0" + str(i) + ".txt", "r") as myfile:
            lines = myfile.readlines()
            line = 0
            features = np.zeros((lines.__len__(), 5), dtype=np.float)
            for l in lines:
                feats = []

                a = l.replace("\n", "").split(" ")

                t = float(a[0])
                x = float(a[1])
                y = float(a[2])
                pressure = float(a[3])

                if line == 0:
                    vx = 0
                    vy = 0
                else:
                    vx = float((x - previous['x']) / (t - previous['t']))
                    vy = float((y - previous['y']) / (t - previous['t']))

                previous['t'] = t
                previous['x'] = x
                previous['y'] = y

                feats.extend((x, y, vx, vy, pressure))
                features[line] = feats
                line = line + 1
        enrollment[filenum + "-" + str(i)] = features

verification = {}

for filename in os.listdir("verification/"):
    with open("verification/" + filename, "r") as myfile:
        lines = myfile.readlines()
        linev = 0
        features = np.zeros((lines.__len__(), 5), dtype=np.float)
        for l in lines:
            feats = []

            a = l.replace("\n", "").split(" ")

            t = float(a[0])
            x = float(a[1])
            y = float(a[2])
            pressure = float(a[3])

            if linev == 0:
                vx = 0
                vy = 0
            else:
                vx = float((x - previous['x']) / (t - previous['t']))
                vy = float((y - previous['y']) / (t - previous['t']))

            previous['t'] = t
            previous['x'] = x
            previous['y'] = y

            feats.extend((x, y, vx, vy, pressure))
            features[linev] = feats
            linev = linev + 1
    verification[filename.replace(".txt", "")] = features

#Load transcriptions
transcriptions = {}
with open("gt.txt", "r") as myfile:
    lines = myfile.readlines()
    for l in lines:
        a = l.replace("\n", "").split(" ")
        transcriptions[a[0]] = a[1]

print("Start: " + str(datetime.datetime.now().time()))


mean_precisions = []
res = {}

#compute dissimilarity for each verification signature wrt the 5 genuine ones
for signature in verification:
    dists = []
    for genuine in enrollment:
        dist, path = fastdtw(verification[signature], enrollment[genuine], dist=euclidean)
        dists.append(dist)

    if len(dists) > 0:
        res[signature] = np.mean(dists)

# Sort elements by increasing distance
results = sorted(res, key=res.get, reverse=False)

print("End of dtw: " + str(datetime.datetime.now().time()))

pr = [transcriptions[r] for r in results[:10]]
print(pr)

# # Compute average precision
# precisions = []
# tp = 0
# fp = 0
#
# for r in results:
#     if transcriptions[r] == transcriptions[signature]:
#         tp = tp + 1
#     else:
#         fp = fp + 1
#     precisions.append(tp / (tp + fp))
#
# if len(precisions) > 0:
#     if np.mean(precisions) > 0:
#         mean_precisions.append(np.mean(precisions))
#
#     print("avg_precision: " + str(np.mean(precisions)))
#
# print("ratio")
# print("Average mean precision: " + str(np.mean(mean_precisions)))
# print("End: " + str(datetime.datetime.now().time()))
