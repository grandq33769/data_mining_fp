import tensorflow as tf;
import numpy as np;
import csv;
import dexJS as JS;
import random;
from sklearn.cluster import KMeans;
import math;
import itertools;

print("程序開始﹗");

# 訓練資料佔總
TRAINING_RATIO = 0.8;
# 資料來源 (使用 testAnalysis.py 的結果)
DATASET = "tempData5.csv";
# 理想 R2
THRESHOLD = 0.5;


# ------------------- 第一步 -------------------
# 資料讀取：
# 利用 readCSV() 擷取資料，放在 movieHeader 及 movieData 兩個變數中
print("第一步: 資料讀取...");

# CSV 讀取，以 JS.Array 規格，回傳標題列及資料列
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

movieHeader, movieData = readCSV(DATASET, True);


# ------------------- 第二步 -------------------
# 後期資料前處理：
# 這部分的處理是針對這次再讀進資料的一些處理
# 包括，把文字變回數字、進行正規化、加入多項式、加入邏輯迴歸的分類等

def getColNo(colName):
    return movieHeader.indexOf(colName);

# ------------------- 第二步 - A -------------------
# 修改欄位：
# 下列會針對不同屬性進行修改
# colFtnAry 內為每項：[[屬性名稱列表], lambda 函數]
# adult 會把 TRUE 變成 1 , FALSE 變成 0
# Jan, Feb, Mar,... Spoken Language-Seg-3 會從文字修正成 int 整數
# budget, revenue, runtime 會從文字修正成 flaot 數字
# runtime 會從文字修正成 flaot 數字，如有空格則修正成 0
print("第二步 - A: 修改欄位...");

# modifyColumn(): 把所有資料項目進行更新
def modifyColumn(data, header, colFtnAry):
    colFtnAry = colFtnAry.map(lambda ele: [ele[0].map(lambda headerName: header.indexOf(headerName)), ele[1]]);
    def replaceContent(row, colID, ftn):
        row[colID] = ftn(row, colID);
    data.forEach(lambda row,idx,ary: colFtnAry.forEach(lambda colFtn, idx2, ary2: colFtn[0].forEach(lambda colID, idx3, ary3: replaceContent(row, colID, colFtn[1]))));

modifyColumn(movieData, movieHeader, JS.Array([[JS.Array(["adult"]), lambda row,colID: 1 if row[colID] == "TRUE" else 0],
                                              [JS.Array(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","year","month","vote_count","year",
                                                "month","Cast_121323_Bess Flowers","Cast_113_Christopher Lee","Cast_4165_John Wayne","Cast_2231_Samuel L. Jackson",
                                                "Cast_3895_Michael Caine","Cast-Seg-1","Cast-Seg-2","Cast-Seg-3","Cast-Seg-4","Crew_9062_Cedric Gibbons",
                                                "Crew_2952_Avy Kaufman","Crew_4350_Edith Head","Crew_102429_Roger Corman","Crew_1259_Ennio Morricone",
                                                "Crew-Seg-1","Crew-Seg-2","Crew-Seg-3","Crew-Seg-4","Original Language_en_en","Original Language_fr_fr",
                                                "Original Language-Seg-1","Original Language-Seg-2","Original Language-Seg-3","Genre_18_Drama","Genre_35_Comedy",
                                                "Genre_53_Thriller","Genre_10749_Romance","Genre_28_Action","Genre-Seg-1","Genre-Seg-2","Production Company_6194_Warner Bros.",
                                                "Production Company_8411_Metro-Goldwyn-Mayer (MGM)","Production Company_4_Paramount Pictures",
                                                "Production Company_306_Twentieth Century Fox Film Corporation","Production Company_33_Universal Pictures",
                                                "Production Company-Seg-1","Production Company-Seg-2","Production Company-Seg-3","Production Company-Seg-4",
                                                "Production Country_US_United States of America","Production Country_GB_United Kingdom","Production Country_FR_France",
                                                "Production Country_DE_Germany","Production Country_IT_Italy","Production Country-Seg-1","Production Country-Seg-2",
                                                "Production Country-Seg-3","Spoken Language_en_English","Spoken Language-Seg-1","Spoken Language-Seg-2","Spoken Language-Seg-3"]), lambda row,colID: int(row[colID])],
                                             [JS.Array(["revenue","budget","popularity"]), lambda row,colID: float(row[colID])],
                                             [JS.Array(["runtime"]), lambda row,colID: float(row[colID]) if row[colID] else 0]]));
modifyColumn(movieData, movieHeader, JS.Array([[JS.Array(["budget"]), lambda row,colID: row[colID] if row[colID] >= 1000 else None]]));


def tryKMeans(data, columnName1, columnName2, trialCount = 10, n_cluster = 12):
    print("KMeans 資料準備中...");
    colNo1, colNo2 = getColNo(columnName1), getColNo(columnName2);
    preX = data.map(lambda row: JS.Array([row[colNo1], row[colNo2]]));
    preX = preX.filter(lambda row: row[0] is not None and row[1] is not None);
    preX = np.array(preX);
    
    
    with open('KMeans - '+columnName1+", "+ columnName2+'.csv', 'w', newline='') as csvfile:
        toWriter = csv.writer(csvfile);
        toWriter.writerow(["Trial", "Prediction Score"]);
        
    for i in range(0, trialCount):
        print("KMeans 訓練及預測 - " + str(i) + "...");
        np.random.shuffle(preX);
        trainProp = round(len(preX) * TRAINING_RATIO);
        trainX = preX[0:trainProp];
        testX = preX[trainProp:len(preX)];
        kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(trainX);
        testPrediction = kmeans.score(testX);
        predX = kmeans.predict(testX);
        with open('KMeans - '+columnName1+", "+ columnName2+'.csv', 'a', newline='') as csvfile:
            toWriter = csv.writer(csvfile);
            toWriter.writerow([str(i), str(testPrediction)]);
        with open('KMeans - '+columnName1+", "+ columnName2+' - Visual - ' + str(i) + '.csv', 'a', newline='') as csvfile:
            toWriter = csv.writer(csvfile);
            toWriter.writerow([columnName1, columnName2, "Prediction Score"]);
            for i,row in enumerate(testX):
                toWriter.writerow([*row, str(predX[i])]);

tryKMeans(movieData, "revenue", "budget");
