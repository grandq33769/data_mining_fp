import csv;
import asyncio;
import dexJS as JS;
import sys;
import re;
import numpy as np;
from sklearn.cluster import KMeans;

def readCSV(dataPath, hasHeader = False):
    with open(dataPath, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f);
        data = JS.Array(); header = JS.Array();
        if (hasHeader):
            for i, row in enumerate(reader):
                if (i>0): data.push(JS.Array(row).map(lambda cell: JS.String(cell)));
                else: header = JS.Array([JS.String(cell) for cell in row]);
        else:
            data = JS.Array([JS.Array(row).map(lambda cell: JS.String(cell)) for row in reader]);
        return header, data;

def tableToObject(jsarray, header, JSONColNames):
    return jsarray.map(lambda row: JS.Map([([header[i], cell] if not JSONColNames.includes(header[i]) else [header[i], JS.AST.parse(cell)]) for i,cell in enumerate(row)]));

print("Reading Movie Data...");
movieHeader, movieData = readCSV("movies_metadata (lineBreakFixed).csv", True);
print("Reading Credit Data...");
creditHeader, creditData = readCSV("credits.csv", True);

print("Converting Data into objects - Movie...");
movieData = tableToObject(movieData, movieHeader, JS.Array(["belongs_to_collection", "genres", "production_companies", "production_countries", "spoken_languages"]));
# this part may be slow because many long strings need to be parsed
print("Converting Data into objects - Credit...");
creditData = tableToObject(creditData, creditHeader, JS.Array(["cast", "crew"]));

print("Analysis starts");


preX = movieData.map(lambda row: JS.Array([float(row.get("budget")), float(row.get("revenue"))]));
preX = np.array(preX);
np.random.shuffle(preX);
trainProp = round(len(preX) * .75);
trainX = preX[0:trainProp];
testX = preX[trainProp:len(preX)];

kmeans = KMeans(n_clusters=12, random_state=0).fit(trainX);
testPrediction = kmeans.predict(testX);
with open('kmeansTest.csv', 'w', newline='') as csvfile:
    toWriter = csv.writer(csvfile);
    JS.Array(trainX).forEach(lambda row, i, ary: toWriter.writerow(r+ow));
