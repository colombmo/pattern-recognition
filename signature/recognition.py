import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import datetime
import os

# Read feature vectors from .txt
features = {}
enrollment = {}
previous = {}

# Get name of signatures files
with open("users.txt", "r") as myfile:
    lines = myfile.readlines()

#Loading enrollment data(genuine signatures)
for fn in lines:
    filenum = fn.replace("\n", "")
    enrollment[filenum]={}
    for i in range(1, 6):
        with open("enrollment/" + filenum + "-g-%02d.txt" % (i,), "r") as myfile:
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
        enrollment[filenum][i] = features

#Loading verification data
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

#compute the mean distance between variations of genuine signatures for each rider
mean_dist = {}
for author in enrollment:
    mean_dist[author] = {}
    dists = []
    for var_i in enrollment[author]:
        for var_j in enrollment[author]:
            if int(var_i)<int(var_j):
                dist, path = fastdtw(enrollment[author][var_i], enrollment[author][var_j], dist=euclidean)
                dists.append(dist)
    mean_dist[author] = np.mean(dists)

res = {}
predictions={}
threshold = 10000

#compute dissimilarity for each verification signature wrt the 5 genuine ones
for signature in verification:
    predictions[signature]={}
    dists = []
    tempAuthor = signature.split("-")[0]
    for genuine in enrollment[tempAuthor]:
        dist, path = fastdtw(verification[signature], enrollment[tempAuthor][genuine], dist=euclidean)
        dists.append(dist)

    if len(dists) > 0:
        res[signature] = np.mean(dists)

    if abs(res[signature] - mean_dist[tempAuthor]) < threshold:
        predictions[signature] = 'g'
    else:
        predictions[signature] = 'f'

# Compute average precision
mean_precisions = []
precisions = []
tp = 0
fp = 0

for r in predictions:
    if transcriptions[r] == predictions[r]:
        tp = tp + 1
    else:
        fp = fp + 1
    precisions.append(tp / (tp + fp))

if len(precisions) > 0:
    if np.mean(precisions) > 0:
        mean_precisions.append(np.mean(precisions))

    print("avg_precision: " + str(np.mean(precisions)))

print("ratio")
print("Average mean precision: " + str(np.mean(mean_precisions)))
print("End: " + str(datetime.datetime.now().time()))
