import tensorflow as tf;
import numpy as np;
import csv;
import dexJS as JS;
import random;
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score;
import math;
import itertools;

print("程序開始﹗");

# 訓練資料佔總
TRAINING_RATIO = 0.8;
# 資料來源 (使用 testAnalysis.py 的結果)
DATASET = "tempData5.csv";
# 理想 R2
THRESHOLD = 0.3;


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

# ------------------- 第二步 - B -------------------
# 將會新增欄位：
# revenueNorm, revenueLog => 從原本的 revenue 引伸出來
# budgetSquare, runtimeSquare => 從原本的 budget/runtime 套用二次函數
print("第二步 - B: 新增欄位...");
def createColumn(data, header, colFtnAry):
    header.push(*colFtnAry.map(lambda colFtn: colFtn[0]));
    data.forEach(lambda row,idx,ary: colFtnAry.forEach(lambda colFtn, idx2, ary2: row.push(colFtn[1](row))));

createColumn(movieData, movieHeader, JS.Array([["revenueNorm", lambda row: row[getColNo("revenue")]],
                                               ["revenueLog", lambda row: math.log(row[getColNo("revenue")])],
                                               ["budgetLog", lambda row: math.log(row[getColNo("budget")]) if row[getColNo("budget")] is not None else None],
                                               ["budgetSquare", lambda row: row[getColNo("budget")]**2 if row[getColNo("budget")] is not None else None],
                                               ["runtimeSquare", lambda row: row[getColNo("runtime")]**2]
                                             
]));

# ------------------- 第二步 - C -------------------
# 將會新增欄位：
# revenueNorm, revenueLog => 從原本的 revenue 引伸出來
# budgetSquare, runtimeSquare => 從原本的 budget/runtime 套用二次函數
print("第二步 - C: 處理正規化 Normalization...");
def normalization(data, header, colNames):
    colInfo = JS.Map(colNames.map(lambda colName: [colName, header.indexOf(colName)]));
    colInfo.forEach(lambda colID,colName,m: m.set(colName, [colID, min(*movieData.map(lambda row: row[colID]).filter(lambda v: v is not None)), max(*movieData.map(lambda row: row[colID]).filter(lambda v: v is not None))]));
    def normalizeValue(row, colInfo):
        row[colInfo[0]] = (row[colInfo[0]] - colInfo[1]) / (colInfo[2] - colInfo[1]) if row[colInfo[0]] is not None else None;
    data.forEach(lambda row,idx,ary: colInfo.forEach(lambda colInfo, colName, m: normalizeValue(row, colInfo)));
    return colInfo;

colInfo = normalization(movieData, movieHeader, JS.Array(["runtime", "budget", "revenueNorm", "revenueLog", "popularity"]));

# ------------------- 第二步 - X -------------------
# 將會新增欄位：
# revPop => 擷取 revenue:popularity
print("第二步 - X: 新增欄位 revPop...");
def createColumn(data, header, colFtnAry):
    header.push(*colFtnAry.map(lambda colFtn: colFtn[0]));
    data.forEach(lambda row,idx,ary: colFtnAry.forEach(lambda colFtn, idx2, ary2: row.push(colFtn[1](row))));

createColumn(movieData, movieHeader, JS.Array([["revPop", lambda row: math.log((max(row[getColNo("revenueNorm")], 0.000001))/(max(row[getColNo("popularity")],0.000001)))]
                                              ]));

# ------------------- 第二步 - D -------------------
# 將會新增欄位：
# revenue 會建立為 10 個 class，按照排序分割
print("第二步 - D: Dependent Variable 分類...");
def createClassByRank(data, header, colFtnAry):
    colFtnAry.forEach(lambda colNo, idx, ary: header.push(colNo[0]+" (Class)"));
    colFtnAry = colFtnAry.map(lambda headerName: [header.indexOf(headerName[0]), headerName[1]]);
    dataLength = data.length;
    for x in colFtnAry:
        sortValues = data.map(lambda row: row[x[0]]).sort(lambda a,b: 1 if a > b else -1);
        classSize = round(dataLength / x[1]);
        x[1] = JS.Array([[(((sortValues[(b-1)*classSize-1] + sortValues[(b-1)*classSize]) / 2) if b > 1 else sortValues[0]),((sortValues[b*classSize-1] + sortValues[b*classSize]) / 2)] for b in range(1, x[1])]);
        
    data.forEach(lambda row,idx,ary: colFtnAry.forEach(lambda colFtn, idx2, ary2: row.push(colFtn[1].findIndex(lambda toRange: row[colFtn[0]] >= toRange[0] and row[colFtn[0]]<=toRange[1])+1)));
        
colInfo = createClassByRank(movieData, movieHeader, JS.Array([["revenue", 8], ["revPop", 3]]));

# ------------------- 第三步 -------------------
# 資料分析：
# 透過 Auto-Regression 把所有有可能的模型，進行推測
# 把各自的 Prediction / R-Square 運算出來，
# 若然 R-Square 很高 (>=0.7)，把所有 Test Data 匯出，仔細瞭解預測數字
print("第三步: 資料分析...");

# Logistic Regression 函數
# 使用 sklearn 進行 logistic regression
def regression(dataset, filename, trialName, featureColNames, outputColName, outputColNameNormalized = "", name="", tfSteps=1000):
    print("\n\n迴歸分析:", name);
    random.shuffle(dataset, random.seed());
    trainingCount = round(TRAINING_RATIO*dataset.length);
    trainData = dataset.slice(0,trainingCount);
    testData = dataset.slice(trainingCount);

    
    colX = [movieHeader.indexOf(i) for i in featureColNames];
    colY = movieHeader.indexOf(outputColName);
    
    trainY = trainData.map(lambda row: row[colY]);
    trainX = trainData.map(lambda row: [row[i] for i in colX]);
    
    testY = testData.map(lambda row: row[colY]);
    testX = testData.map(lambda row: [row[i] for i in colX]);

    outputInfo = JS.Array([trialName, name]);

    regr = linear_model.LogisticRegression();
    regr.fit(trainX, trainY);
    predY = regr.predict(testX);
    print("\nSK Learn: ");
    scoreSK = regr.score(testX, testY._itr);
    print("預測成功率 (Score): ", scoreSK);
    outputInfo.push(scoreSK);
    if (scoreSK >= THRESHOLD):
        with open(filename + "-Result-" + trialName + ".csv", 'w', newline='', encoding='utf-8-sig') as csvfile:
             toWriter = csv.writer(csvfile);
             toWriter.writerow([*featureColNames, outputColName, "Prediction (SK)"]);
             for idx, row in enumerate(testX):
                   toWriter.writerow([*row, testY[idx], predY[idx]]);
    return outputInfo;


def regressionHandler(movieData, filename, trialName, targetClass, inputs):
    outputTable = JS.Array();
    for x in inputs:
        outputInfo = regression(movieData, filename, trialName, x, targetClass, name=JS.Array(x).join(",")+" -> " + targetClass);
        outputTable.push(outputInfo);
    with open(filename, 'a', newline='', encoding='utf-8-sig') as csvfile:
        toWriter = csv.writer(csvfile);
        for row in outputTable:
            toWriter.writerow(row);
    return outputTable;

def autoRegressionHandler(movieData, filename, targetClass, inputDict, inputFilterDict, mustInputDict = {}):
    print("準備分析...");
    trialID = 0;
    outputTable = JS.Array();
    filename = filename+'.csv';
    allMust = [];
    for k,attrs in mustInputDict.items():
        allMust[len(allMust):] = attrs;
    print("創建檔案...");
    with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        toWriter = csv.writer(csvfile);
        toWriter.writerow(["Trial", "Model", "SK Prediction Score"]);

    for l in range(1,len(inputDict)+1):
            for x in itertools.combinations(inputDict.keys(), l):
                allAttr = [*allMust];
                for attrGroup in x:
                    allAttr[len(allAttr):] = inputDict[attrGroup];
                toTestData = movieData;
                for attr in allAttr:
                    if (attr in inputFilterDict):
                        colID = getColNo(attr); lambdaFtn = inputFilterDict[attr];
                        toTestData = toTestData.filter(lambda row: lambdaFtn(row[colID]));
                outputTable.push(*regressionHandler(toTestData, filename, str(trialID), targetClass, [allAttr]));
                trialID = trialID + 1;


# -------------- 自動「暴力法」建設 --------------
# 1a: 基本模型 => revenue
autoRegressionHandler(movieData, "logisticRegressionResults-revenue-basic","revenue (Class)",
                      {"year": ["year"], "budget": ["budget"], "runtime": ["runtime"],
                       "cast": ["Cast_121323_Bess Flowers","Cast_113_Christopher Lee","Cast_4165_John Wayne","Cast_2231_Samuel L. Jackson","Cast_3895_Michael Caine","Cast-Seg-1","Cast-Seg-2","Cast-Seg-3","Cast-Seg-4"],
                       "crew": ["Crew_9062_Cedric Gibbons","Crew_2952_Avy Kaufman","Crew_4350_Edith Head","Crew_102429_Roger Corman","Crew_1259_Ennio Morricone","Crew-Seg-1","Crew-Seg-2","Crew-Seg-3","Crew-Seg-4"],
                       "genre": ["Genre_18_Drama","Genre_35_Comedy","Genre_53_Thriller","Genre_10749_Romance","Genre_28_Action","Genre-Seg-1","Genre-Seg-2"],
                       "production_company": ["Production Company_6194_Warner Bros.","Production Company_8411_Metro-Goldwyn-Mayer (MGM)","Production Company_4_Paramount Pictures","Production Company_306_Twentieth Century Fox Film Corporation","Production Company_33_Universal Pictures"],
                       "month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"],
                       "production_country": ["Production Country_US_United States of America","Production Country_GB_United Kingdom","Production Country_FR_France","Production Country_DE_Germany","Production Country_IT_Italy","Production Country-Seg-1","Production Country-Seg-2","Production Country-Seg-3"]
                       },
                      {"budget": lambda val: val is not None,
                       "runtime": lambda val: val > 0});

# 1b: 基本模型, 固定使用 rumtime-square => revenue
autoRegressionHandler(movieData, "logisticRegressionResults-revenue-runtimeSquare","revenue (Class)",
                      {"year": ["year"], "budget": ["budget"],
                       "cast": ["Cast_121323_Bess Flowers","Cast_113_Christopher Lee","Cast_4165_John Wayne","Cast_2231_Samuel L. Jackson","Cast_3895_Michael Caine","Cast-Seg-1","Cast-Seg-2","Cast-Seg-3","Cast-Seg-4"],
                       "crew": ["Crew_9062_Cedric Gibbons","Crew_2952_Avy Kaufman","Crew_4350_Edith Head","Crew_102429_Roger Corman","Crew_1259_Ennio Morricone","Crew-Seg-1","Crew-Seg-2","Crew-Seg-3","Crew-Seg-4"],
                       "genre": ["Genre_18_Drama","Genre_35_Comedy","Genre_53_Thriller","Genre_10749_Romance","Genre_28_Action","Genre-Seg-1","Genre-Seg-2"],
                       "production_company": ["Production Company_6194_Warner Bros.","Production Company_8411_Metro-Goldwyn-Mayer (MGM)","Production Company_4_Paramount Pictures","Production Company_306_Twentieth Century Fox Film Corporation","Production Company_33_Universal Pictures"],
                       "month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"],
                       "production_country": ["Production Country_US_United States of America","Production Country_GB_United Kingdom","Production Country_FR_France","Production Country_DE_Germany","Production Country_IT_Italy","Production Country-Seg-1","Production Country-Seg-2","Production Country-Seg-3"]
                       },
                      {"budget": lambda val: val is not None,
                       "runtimeSquare": lambda val: val > 0},
                      {"runtimeSquare": ["runtimeSquare"]});

# 1c: 基本模型, 固定使用 rumtime-square, budget-log => revenue
autoRegressionHandler(movieData, "logisticRegressionResults-revenue-budgetLog&runtimeSquare","revenue (Class)",
                      {"year": ["year"], 
                       "cast": ["Cast_121323_Bess Flowers","Cast_113_Christopher Lee","Cast_4165_John Wayne","Cast_2231_Samuel L. Jackson","Cast_3895_Michael Caine","Cast-Seg-1","Cast-Seg-2","Cast-Seg-3","Cast-Seg-4"],
                       "crew": ["Crew_9062_Cedric Gibbons","Crew_2952_Avy Kaufman","Crew_4350_Edith Head","Crew_102429_Roger Corman","Crew_1259_Ennio Morricone","Crew-Seg-1","Crew-Seg-2","Crew-Seg-3","Crew-Seg-4"],
                       "genre": ["Genre_18_Drama","Genre_35_Comedy","Genre_53_Thriller","Genre_10749_Romance","Genre_28_Action","Genre-Seg-1","Genre-Seg-2"],
                       "production_company": ["Production Company_6194_Warner Bros.","Production Company_8411_Metro-Goldwyn-Mayer (MGM)","Production Company_4_Paramount Pictures","Production Company_306_Twentieth Century Fox Film Corporation","Production Company_33_Universal Pictures"],
                       "month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"],
                       "production_country": ["Production Country_US_United States of America","Production Country_GB_United Kingdom","Production Country_FR_France","Production Country_DE_Germany","Production Country_IT_Italy","Production Country-Seg-1","Production Country-Seg-2","Production Country-Seg-3"]
                       },
                      {"budgetLog": lambda val: val is not None,
                       "runtimeSquare": lambda val: val > 0},
                      {"budgetLog": ["budgetLog"], "runtimeSquare": ["runtimeSquare"]});

# 1d: 基本模型, 固定使用 rumtime-square, budget-log => revenue
autoRegressionHandler(movieData, "logisticRegressionResults-revenue-budgetSquare&runtimeSquare","revenue (Class)",
                      {"year": ["year"], 
                       "cast": ["Cast_121323_Bess Flowers","Cast_113_Christopher Lee","Cast_4165_John Wayne","Cast_2231_Samuel L. Jackson","Cast_3895_Michael Caine","Cast-Seg-1","Cast-Seg-2","Cast-Seg-3","Cast-Seg-4"],
                       "crew": ["Crew_9062_Cedric Gibbons","Crew_2952_Avy Kaufman","Crew_4350_Edith Head","Crew_102429_Roger Corman","Crew_1259_Ennio Morricone","Crew-Seg-1","Crew-Seg-2","Crew-Seg-3","Crew-Seg-4"],
                       "genre": ["Genre_18_Drama","Genre_35_Comedy","Genre_53_Thriller","Genre_10749_Romance","Genre_28_Action","Genre-Seg-1","Genre-Seg-2"],
                       "production_company": ["Production Company_6194_Warner Bros.","Production Company_8411_Metro-Goldwyn-Mayer (MGM)","Production Company_4_Paramount Pictures","Production Company_306_Twentieth Century Fox Film Corporation","Production Company_33_Universal Pictures"],
                       "month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"],
                       "production_country": ["Production Country_US_United States of America","Production Country_GB_United Kingdom","Production Country_FR_France","Production Country_DE_Germany","Production Country_IT_Italy","Production Country-Seg-1","Production Country-Seg-2","Production Country-Seg-3"]
                       },
                      {"budgetSquare": lambda val: val is not None,
                       "runtimeSquare": lambda val: val > 0},
                      {"budgetSquare": ["budgetSquare"], "runtimeSquare": ["runtimeSquare"]})


# 2a: 基本模型 => revpop
autoRegressionHandler(movieData, "logisticRegressionResults-revPop-basic","revPop (Class)",
                      {"year": ["year"], "budget": ["budget"], "runtime": ["runtime"],
                       "cast": ["Cast_121323_Bess Flowers","Cast_113_Christopher Lee","Cast_4165_John Wayne","Cast_2231_Samuel L. Jackson","Cast_3895_Michael Caine","Cast-Seg-1","Cast-Seg-2","Cast-Seg-3","Cast-Seg-4"],
                       "crew": ["Crew_9062_Cedric Gibbons","Crew_2952_Avy Kaufman","Crew_4350_Edith Head","Crew_102429_Roger Corman","Crew_1259_Ennio Morricone","Crew-Seg-1","Crew-Seg-2","Crew-Seg-3","Crew-Seg-4"],
                       "genre": ["Genre_18_Drama","Genre_35_Comedy","Genre_53_Thriller","Genre_10749_Romance","Genre_28_Action","Genre-Seg-1","Genre-Seg-2"],
                       "production_company": ["Production Company_6194_Warner Bros.","Production Company_8411_Metro-Goldwyn-Mayer (MGM)","Production Company_4_Paramount Pictures","Production Company_306_Twentieth Century Fox Film Corporation","Production Company_33_Universal Pictures"],
                       "month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"],
                       "production_country": ["Production Country_US_United States of America","Production Country_GB_United Kingdom","Production Country_FR_France","Production Country_DE_Germany","Production Country_IT_Italy","Production Country-Seg-1","Production Country-Seg-2","Production Country-Seg-3"]
                       },
                      {"budget": lambda val: val is not None,
                       "runtime": lambda val: val > 0});

# 2b: 基本模型, 固定使用 rumtime-square, budget-log => revpop
autoRegressionHandler(movieData, "logisticRegressionResults-revPop-runtimeSquare","revPop (Class)",
                      {"year": ["year"], "budget": ["budget"],
                       "cast": ["Cast_121323_Bess Flowers","Cast_113_Christopher Lee","Cast_4165_John Wayne","Cast_2231_Samuel L. Jackson","Cast_3895_Michael Caine","Cast-Seg-1","Cast-Seg-2","Cast-Seg-3","Cast-Seg-4"],
                       "crew": ["Crew_9062_Cedric Gibbons","Crew_2952_Avy Kaufman","Crew_4350_Edith Head","Crew_102429_Roger Corman","Crew_1259_Ennio Morricone","Crew-Seg-1","Crew-Seg-2","Crew-Seg-3","Crew-Seg-4"],
                       "genre": ["Genre_18_Drama","Genre_35_Comedy","Genre_53_Thriller","Genre_10749_Romance","Genre_28_Action","Genre-Seg-1","Genre-Seg-2"],
                       "production_company": ["Production Company_6194_Warner Bros.","Production Company_8411_Metro-Goldwyn-Mayer (MGM)","Production Company_4_Paramount Pictures","Production Company_306_Twentieth Century Fox Film Corporation","Production Company_33_Universal Pictures"],
                       "month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"],
                       "production_country": ["Production Country_US_United States of America","Production Country_GB_United Kingdom","Production Country_FR_France","Production Country_DE_Germany","Production Country_IT_Italy","Production Country-Seg-1","Production Country-Seg-2","Production Country-Seg-3"]
                       },
                      {"budget": lambda val: val is not None,
                       "runtimeSquare": lambda val: val > 0},
                      {"runtimeSquare": ["runtimeSquare"]});

# 2c: 基本模型, 固定使用 rumtime-square, budget-log => revpop
autoRegressionHandler(movieData, "logisticRegressionResults-revPop-budgetLog&runtimeSquare","revPop (Class)",
                      {"year": ["year"], 
                       "cast": ["Cast_121323_Bess Flowers","Cast_113_Christopher Lee","Cast_4165_John Wayne","Cast_2231_Samuel L. Jackson","Cast_3895_Michael Caine","Cast-Seg-1","Cast-Seg-2","Cast-Seg-3","Cast-Seg-4"],
                       "crew": ["Crew_9062_Cedric Gibbons","Crew_2952_Avy Kaufman","Crew_4350_Edith Head","Crew_102429_Roger Corman","Crew_1259_Ennio Morricone","Crew-Seg-1","Crew-Seg-2","Crew-Seg-3","Crew-Seg-4"],
                       "genre": ["Genre_18_Drama","Genre_35_Comedy","Genre_53_Thriller","Genre_10749_Romance","Genre_28_Action","Genre-Seg-1","Genre-Seg-2"],
                       "production_company": ["Production Company_6194_Warner Bros.","Production Company_8411_Metro-Goldwyn-Mayer (MGM)","Production Company_4_Paramount Pictures","Production Company_306_Twentieth Century Fox Film Corporation","Production Company_33_Universal Pictures"],
                       "month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"],
                       "production_country": ["Production Country_US_United States of America","Production Country_GB_United Kingdom","Production Country_FR_France","Production Country_DE_Germany","Production Country_IT_Italy","Production Country-Seg-1","Production Country-Seg-2","Production Country-Seg-3"]
                       },
                      {"budgetLog": lambda val: val is not None,
                       "runtimeSquare": lambda val: val > 0},
                      {"budgetLog": ["budgetLog"], "runtimeSquare": ["runtimeSquare"]});

# 2d: 基本模型, 固定使用 rumtime-square, budget-log => revpop
autoRegressionHandler(movieData, "logisticRegressionResults-revPop-budgetSquare&runtimeSquare","revPop (Class)",
                      {"year": ["year"], 
                       "cast": ["Cast_121323_Bess Flowers","Cast_113_Christopher Lee","Cast_4165_John Wayne","Cast_2231_Samuel L. Jackson","Cast_3895_Michael Caine","Cast-Seg-1","Cast-Seg-2","Cast-Seg-3","Cast-Seg-4"],
                       "crew": ["Crew_9062_Cedric Gibbons","Crew_2952_Avy Kaufman","Crew_4350_Edith Head","Crew_102429_Roger Corman","Crew_1259_Ennio Morricone","Crew-Seg-1","Crew-Seg-2","Crew-Seg-3","Crew-Seg-4"],
                       "genre": ["Genre_18_Drama","Genre_35_Comedy","Genre_53_Thriller","Genre_10749_Romance","Genre_28_Action","Genre-Seg-1","Genre-Seg-2"],
                       "production_company": ["Production Company_6194_Warner Bros.","Production Company_8411_Metro-Goldwyn-Mayer (MGM)","Production Company_4_Paramount Pictures","Production Company_306_Twentieth Century Fox Film Corporation","Production Company_33_Universal Pictures"],
                       "month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"],
                       "production_country": ["Production Country_US_United States of America","Production Country_GB_United Kingdom","Production Country_FR_France","Production Country_DE_Germany","Production Country_IT_Italy","Production Country-Seg-1","Production Country-Seg-2","Production Country-Seg-3"]
                       },
                      {"budgetSquare": lambda val: val is not None,
                       "runtimeSquare": lambda val: val > 0},
                      {"budgetSquare": ["budgetSquare"], "runtimeSquare": ["runtimeSquare"]});
