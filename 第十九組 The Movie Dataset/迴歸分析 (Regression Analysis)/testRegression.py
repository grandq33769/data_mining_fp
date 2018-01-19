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

# ------------------- 第三步 -------------------
# 資料分析：
# 透過 Auto-Regression 把所有有可能的模型，進行推測
# 把各自的 Prediction / R-Square 運算出來，
# 若然 R-Square 很高 (>=0.7)，把所有 Test Data 匯出，仔細瞭解預測數字
print("第三步: 資料分析...");

# ---------------- regression() ----------------
# 從資料集中，指定要測試的項目及模成組成，進行 Regression 分析
# 提供兩種訓練方法：Tensorflow / SK-Learn
# 部分 y 可能是修正後(如正規化/對數)，需提供其還原函數 lambda function，以進行準確度測試
def regression(dataset, filename, trialName, featureColNames, oriOutputCol, outputCol = "", name="", tfSteps=1000, recoverFtn = lambda x:x, useTF = True):
    print("\n\n迴歸分析:", name);
    print("資料準備中...");
    # -------- 資料準備 --------
    # 1. 所有資料會重新隨機排序
    # 2. 分割 Training / Testing 訓練/測試資料
    random.shuffle(dataset, random.seed());
    trainingCount = round(TRAINING_RATIO*dataset.length);
    trainData = dataset.slice(0,trainingCount);
    testData = dataset.slice(trainingCount);

    # 3. 檢查是否需要還原正規化/轉化值
    isOutputAdjusted =bool(len(oriOutputCol)>0) and outputCol != oriOutputCol;

    # 4. 擷取資料集，並獨立分隔 Train/Test 資料，及是否修改的 y
    colX = [movieHeader.indexOf(i) for i in featureColNames];
    colY = movieHeader.indexOf(oriOutputCol);
    colYAdj = movieHeader.indexOf(outputCol) if isOutputAdjusted else None ;
    
    trainY = trainData.map(lambda row: [row[colY]]);
    trainX = trainData.map(lambda row: [row[i] for i in colX]);
    trainYAdj = trainData.map(lambda row: [row[colYAdj]]) if isOutputAdjusted else None ;
    
    testY = testData.map(lambda row: [row[colY]]);
    testX = testData.map(lambda row: [row[i] for i in colX]);
    testYAdj = testData.map(lambda row: [row[colYAdj]]) if isOutputAdjusted else None ;

    # 5. 初始化部分資料
    # outputInfo 是這次運算的項目，會是一列數據，指出是哪個嘗試和他的結果
    outputInfo = JS.Array([trialName, name]);

    # -------- Tensorflow --------
    # 如使用 Tensorflow 作迴歸分析，處理如下：
    if (useTF):
        print("TensorFlow 處理: ");

        # 方便的變數設定函數定義
        # Weight 會以 Truncated Normal Distribution 初始化，standard deviation = 0.1
        # Bias 會以 0.1 為初始值
        def weight_variable(shape):
          initial = tf.truncated_normal(shape, stddev=0.1)
          return tf.Variable(initial)

        def bias_variable(shape):
          initial = tf.constant(0.1, shape=shape)
          return tf.Variable(initial)

        # 因為 Tensorflow 會跑很多不同的模型，先把其重設
        tf.reset_default_graph();

        # 留意目前有多個 x independent variables，
        # 並設定 y = Wx+b 的架構，
        # 而 actualY 會是代入實際結果的 placeholder
        noOfFeatures = len(colX);
        actualY = tf.placeholder(shape=[None, 1], dtype=np.float32);
        x = tf.placeholder(shape=[None, noOfFeatures], dtype=np.float32);
        W = weight_variable([noOfFeatures, 1]);
        b = bias_variable([1]);
        y = tf.matmul(x, W)+ b;

        # Learning Rate 是變數，會視乎訓練情況而改變，但初始值為 0.01
        # Loss Function 是 RMS ，並以 Adam Optimizer 進行 Gradient Descent
        learningRate = tf.Variable(0.01, trainable=False);
        loss = tf.reduce_mean(tf.squared_difference(y, actualY)); 
        train_step = tf.train.AdamOptimizer(learningRate).minimize(loss);

        with tf.Session() as sess:
            # 訓練先要把所有值正式初始化
            tf.global_variables_initializer().run();

            # 訓練過種中的變數，去判斷是否是停下來或改變 Learning Rate
            # 這次的訓練方式，是以每 100 次訓練的 Loss 改變為基準，
            # 若是 Loss 改變不大，則會決定是否需要調低 Learning Rate，
            # 否則若再降不下 Learning Rate 則會停止訓練
            lastLoss = 0;
            training = 1;
            decLearningRate = 2;
            m = 0;
            while True:
                step, lossout, lrNow = sess.run([train_step, loss, learningRate], feed_dict={x: trainX, actualY: (trainYAdj if isOutputAdjusted else trainY)});
                if m==0:
                    lastLoss = lossout*2;
                m = m + 1;
                if (m%100 == 0):
                    print("訓練步數: ", m ,"; Loss:", lossout, "; Learning Rate:", lrNow);
                    if (abs(lossout - lastLoss)/lastLoss < 0.01):
                        if training == 0:
                            if decLearningRate > 0:
                                decLearningRate = decLearningRate - 1;
                                training = 1;
                                nlr = learningRate.assign(lrNow/2);
                                sess.run([nlr]);
                            else:
                                break;
                        else:
                            training = training - 1;
                    lastLoss = lossout;

            # Tensorflow 測試：
            # 同樣地以 RMS 作為測試基準
            # 但同時就會個測試項目，列出模型訓練結果
            rms = tf.reduce_mean(tf.squared_difference(y, actualY)); 
            predictYTF, rmsValue = (sess.run([y, rms], feed_dict={x: testX, actualY: (testYAdj if isOutputAdjusted else testY)}));
            print("TensorFlow 結果: ");
            print("RMS: ",rmsValue);
            outputInfo.push(rmsValue);
            
            # 同時運算每一個測試項目的差異百分比平均值
            predictYTF=JS.Array(predictYTF).map(lambda s: float(s[0]));
            predictYTFRecovered = predictYTF.map(lambda s: (recoverFtn(s) if isOutputAdjusted else s));
            testY2 = JS.Array(testY).map(lambda s: float(s[0]));
            predictDiffTF = JS.Array();
            predictYTFRecovered.forEach(lambda s,i,ary: predictDiffTF.push(abs(s - testY2[i]) / abs(testY2[i])));
            aep =(sum(predictDiffTF)/predictDiffTF.length*100);
            print("平均誤差百分比: ",aep);
            outputInfo.push(aep);

            # 同時運算 R-Square 值
            r2TF = r2_score(testY2, predictYTFRecovered)
            print("R Square (還原值): ",r2TF);
            outputInfo.push(r2TF);
            if isOutputAdjusted:
                r2TFAdj = r2_score(testYAdj, predictYTF);
                print("R Square (正規化/轉移值): ",r2TFAdj);
                outputInfo.push(r2TFAdj);

    # -------- SK Learn --------
    # SK Learn 模型訓練：
    regr = linear_model.LinearRegression();
    regr.fit(trainX, trainYAdj if isOutputAdjusted else trainY);

    # SK Learn 結果需要先取得預測值，再運算 RMS
    predYSK = regr.predict(testX);
    rmsValue = mean_squared_error(testYAdj if isOutputAdjusted else testY, predYSK);
    print("\nSK Learn 結果: ");
    print("RMS: ", rmsValue);
    outputInfo.push(rmsValue);

    # 同時運算每一個測試項目的差異百分比平均值
    predictDiffSK = JS.Array();
    predYSK = JS.Array(predYSK).map(lambda s: float(s[0]));
    predYSKRecovered = predYSK.map(lambda s: (recoverFtn(s) if isOutputAdjusted else s));
    testY2 = JS.Array(testY).map(lambda s: float(s[0]));
    predYSKRecovered.forEach(lambda s,i,ary: predictDiffSK.push(abs(s - testY2[i]) / abs(testY2[i])));
    aepSK =(sum(predictDiffSK)/predictDiffSK.length*100);
    print("平均誤差百分比: ",aepSK);
    outputInfo.push(aepSK);

    # 同時運算 R-Square 值
    r2SK = r2_score(testY2, predYSKRecovered)
    print("R Square (還原值): ",r2SK);
    outputInfo.push(r2SK);
    if isOutputAdjusted:
        r2SKAdj = r2_score(testYAdj, predYSK);
        print("R Square (正規化/轉移值): ",r2SKAdj);
        outputInfo.push(r2SKAdj);

    # -------- 詳細結果儲存 --------
    # 如 R-Square 在 THRESHOLD 值之上，則把整個測試結果存下來仔細觀察
    if (useTF):
        if (r2TF >= THRESHOLD or r2SK >= THRESHOLD):
            with open(filename + "-Result-" + trialName + ".csv", 'w', newline='', encoding='utf-8-sig') as csvfile:
                toWriter = csv.writer(csvfile);
                toWriter.writerow([*featureColNames, outputCol, "Prediction (TF)", "Prediction (SK)"]);
                for idx, row in enumerate(testX):
                    toWriter.writerow([*row, testY[idx][0], predictYTFRecovered[idx], predYSKRecovered[idx]]);
    else:
        if (r2SK >= THRESHOLD):
            with open(filename + "-Result-" + trialName + ".csv", 'w', newline='', encoding='utf-8-sig') as csvfile:
                toWriter = csv.writer(csvfile);
                toWriter.writerow([*featureColNames, outputCol, "Prediction (SK)"]);
                for idx, row in enumerate(testX):
                    toWriter.writerow([*row, testY[idx][0], predYSKRecovered[idx]]);
    return outputInfo;

# ---------------- regressionHandler() ----------------
# 允許透過 inputs 一次性地建立不同 x 模型
# 所有模型完成後會把所有結果加入 (append) 到指定檔案中
def regressionHandler(movieData, filename, trialName, outputCol, inputs, recoverFtn=lambda x: (x-minRevenue)*(maxRevenue-minRevenue), useTF = True, oriOutputCol = "revenue"):
    outputTable = JS.Array();
    for x in inputs:
        outputInfo = regression(movieData, filename, trialName, x, oriOutputCol, outputCol = outputCol, name=JS.Array(x).join(",")+" -> " + outputCol, recoverFtn=recoverFtn, useTF=useTF);
        outputTable.push(outputInfo);
    with open(filename, 'a', newline='', encoding='utf-8-sig') as csvfile:
        toWriter = csv.writer(csvfile);
        for row in outputTable:
            toWriter.writerow(row);
    return outputTable;

# ---------------- autoRegressionHandler() ----------------
# 自動組合不同參數，傳輸組合的模型至 regressionHandler() 進行訓練及測試
def autoRegressionHandler(movieData, filename, outputCol_RecoverDict, inputDict, inputFilterDict, mustInputDict = {}, useTF = True, oriOutputCol = "revenue"):
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
        if (useTF):
            toWriter.writerow(["Trial", "Model", "TF RMS", "TF Error %", "TF R2", "TF R2 (normalized)", "SK RMS", "SK Error %", "SK R2", "SK R2 (normalized)"]);
        else:
            toWriter.writerow(["Trial", "Model", "SK RMS", "SK Error %", "SK R2", "SK R2 (normalized)"]);
    for k, v in outputCol_RecoverDict.items():
        for l in range(1,len(inputDict)+1):
            # 透過 itertools.combinations ，建立不同參數的模型
            for x in itertools.combinations(inputDict.keys(), l):
                allAttr = [*allMust];
                for attrGroup in x:
                    allAttr[len(allAttr):] = inputDict[attrGroup];
                toTestData = movieData;
                for attr in allAttr:
                    if (attr in inputFilterDict):
                        colID = getColNo(attr); lambdaFtn = inputFilterDict[attr];
                        toTestData = toTestData.filter(lambda row: lambdaFtn(row[colID]));
                outputTable.push(*regressionHandler(toTestData, filename, str(trialID), k, [allAttr], v, useTF = useTF, oriOutputCol = oriOutputCol));
                trialID = trialID + 1;

# -------------- 先導直覺性模型建設 --------------
# 指定模型的參數，透過 regressionHandler()，
# 就 revenueLog 及 revenueNorm 進行模型訓練及測試
with open("preliminaryRegressions-Result-All.csv", 'w', newline='', encoding='utf-8-sig') as csvfile:
   toWriter = csv.writer(csvfile);
   toWriter.writerow(["Trial", "Model", "TF RMS", "TF Error %", "TF R2", "TF R2 (normalized)", "SK RMS", "SK Error %", "SK R2", "SK R2 (normalized)"]);

regressionHandler(movieData, "preliminaryRegressions-Result-All.csv", "All", "revenueLog",
                                    [["month"], ["year"],
                                     ["Cast_121323_Bess Flowers","Cast_113_Christopher Lee","Cast_4165_John Wayne","Cast_2231_Samuel L. Jackson","Cast_3895_Michael Caine","Cast-Seg-1","Cast-Seg-2","Cast-Seg-3","Cast-Seg-4"],
                                     ["Genre_18_Drama","Genre_35_Comedy","Genre_53_Thriller","Genre_10749_Romance","Genre_28_Action","Genre-Seg-1","Genre-Seg-2"],
                                     ["Production Company_6194_Warner Bros.","Production Company_8411_Metro-Goldwyn-Mayer (MGM)","Production Company_4_Paramount Pictures","Production Company_306_Twentieth Century Fox Film Corporation","Production Company_33_Universal Pictures"],
                                     ["month", "year",
                                     "Cast_121323_Bess Flowers","Cast_113_Christopher Lee","Cast_4165_John Wayne","Cast_2231_Samuel L. Jackson","Cast_3895_Michael Caine","Cast-Seg-1","Cast-Seg-2","Cast-Seg-3","Cast-Seg-4",
                                     "Genre_18_Drama","Genre_35_Comedy","Genre_53_Thriller","Genre_10749_Romance","Genre_28_Action","Genre-Seg-1","Genre-Seg-2",
                                     "Production Company_6194_Warner Bros.","Production Company_8411_Metro-Goldwyn-Mayer (MGM)","Production Company_4_Paramount Pictures","Production Company_306_Twentieth Century Fox Film Corporation","Production Company_33_Universal Pictures"]
                                     ],
                                    lambda x: math.exp(x*(colInfo.get("revenueLog")[2]-colInfo.get("revenueLog")[1])+colInfo.get("revenueLog")[1]));

regressionHandler(movieData, "preliminaryRegressions-Result-All.csv", "All", "revenueNorm",
                                    [["month"], ["year"],
                                     ["Cast_121323_Bess Flowers","Cast_113_Christopher Lee","Cast_4165_John Wayne","Cast_2231_Samuel L. Jackson","Cast_3895_Michael Caine","Cast-Seg-1","Cast-Seg-2","Cast-Seg-3","Cast-Seg-4"],
                                     ["Genre_18_Drama","Genre_35_Comedy","Genre_53_Thriller","Genre_10749_Romance","Genre_28_Action","Genre-Seg-1","Genre-Seg-2"],
                                     ["Production Company_6194_Warner Bros.","Production Company_8411_Metro-Goldwyn-Mayer (MGM)","Production Company_4_Paramount Pictures","Production Company_306_Twentieth Century Fox Film Corporation","Production Company_33_Universal Pictures"],
                                     ["month", "year",
                                     "Cast_121323_Bess Flowers","Cast_113_Christopher Lee","Cast_4165_John Wayne","Cast_2231_Samuel L. Jackson","Cast_3895_Michael Caine","Cast-Seg-1","Cast-Seg-2","Cast-Seg-3","Cast-Seg-4",
                                     "Genre_18_Drama","Genre_35_Comedy","Genre_53_Thriller","Genre_10749_Romance","Genre_28_Action","Genre-Seg-1","Genre-Seg-2",
                                     "Production Company_6194_Warner Bros.","Production Company_8411_Metro-Goldwyn-Mayer (MGM)","Production Company_4_Paramount Pictures","Production Company_306_Twentieth Century Fox Film Corporation","Production Company_33_Universal Pictures"]
                                     ],
                                    lambda x: x*(colInfo.get("revenueNorm")[2]-colInfo.get("revenueNorm")[1])+colInfo.get("revenueNorm")[1]);




# -------------- 自動「暴力法」建設 --------------
# 0: 測試項目，以比較少的欄位進行測試
autoRegressionHandler(movieData, "regressionResults-revenue-warmup",
                      {"revenueNorm": lambda x: x*(colInfo.get("revenueNorm")[2]-colInfo.get("revenueNorm")[1])+colInfo.get("revenueNorm")[1],
                       "revenueLog": lambda x: math.exp(x*(colInfo.get("revenueLog")[2]-colInfo.get("revenueLog")[1])+colInfo.get("revenueLog")[1])},
                      {"year": ["year"], "budget": ["budget"], 
                       "cast": ["Cast_121323_Bess Flowers","Cast_113_Christopher Lee","Cast_4165_John Wayne","Cast_2231_Samuel L. Jackson","Cast_3895_Michael Caine","Cast-Seg-1","Cast-Seg-2","Cast-Seg-3","Cast-Seg-4"],
                       "genre": ["Genre_18_Drama","Genre_35_Comedy","Genre_53_Thriller","Genre_10749_Romance","Genre_28_Action","Genre-Seg-1","Genre-Seg-2"],
                       "month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"],
                       },
                      {"budget": lambda val: val is not None,
                       "runtime": lambda val: val > 0}, useTF = True);

# 1a: 基本模型 => revenue
autoRegressionHandler(movieData, "regressionResults-revenue-basic",
                      {"revenueNorm": lambda x: x*(colInfo.get("revenueNorm")[2]-colInfo.get("revenueNorm")[1])+colInfo.get("revenueNorm")[1],
                       "revenueLog": lambda x: math.exp(x*(colInfo.get("revenueLog")[2]-colInfo.get("revenueLog")[1])+colInfo.get("revenueLog")[1])},
                      {"year": ["year"], "budget": ["budget"], "runtime": ["runtime"],
                       "cast": ["Cast_121323_Bess Flowers","Cast_113_Christopher Lee","Cast_4165_John Wayne","Cast_2231_Samuel L. Jackson","Cast_3895_Michael Caine","Cast-Seg-1","Cast-Seg-2","Cast-Seg-3","Cast-Seg-4"],
                       "crew": ["Crew_9062_Cedric Gibbons","Crew_2952_Avy Kaufman","Crew_4350_Edith Head","Crew_102429_Roger Corman","Crew_1259_Ennio Morricone","Crew-Seg-1","Crew-Seg-2","Crew-Seg-3","Crew-Seg-4"],
                       "genre": ["Genre_18_Drama","Genre_35_Comedy","Genre_53_Thriller","Genre_10749_Romance","Genre_28_Action","Genre-Seg-1","Genre-Seg-2"],
                       "production_company": ["Production Company_6194_Warner Bros.","Production Company_8411_Metro-Goldwyn-Mayer (MGM)","Production Company_4_Paramount Pictures","Production Company_306_Twentieth Century Fox Film Corporation","Production Company_33_Universal Pictures"],
                       "month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"],
                       "production_country": ["Production Country_US_United States of America","Production Country_GB_United Kingdom","Production Country_FR_France","Production Country_DE_Germany","Production Country_IT_Italy","Production Country-Seg-1","Production Country-Seg-2","Production Country-Seg-3"]
                       },
                      {"budget": lambda val: val is not None,
                       "runtime": lambda val: val > 0}, useTF = False);

# 1b: 基本模型, 固定使用 rumtime-square => revenue
autoRegressionHandler(movieData, "regressionResults-revenue-runtimeSquare",
                      {"revenueNorm": lambda x: x*(colInfo.get("revenueNorm")[2]-colInfo.get("revenueNorm")[1])+colInfo.get("revenueNorm")[1],
                       "revenueLog": lambda x: math.exp(x*(colInfo.get("revenueLog")[2]-colInfo.get("revenueLog")[1])+colInfo.get("revenueLog")[1])},
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
                      {"runtimeSquare": ["runtimeSquare"]}, useTF = False);

# 1c: 基本模型, 固定使用 rumtime-square, budget-log => revenue
autoRegressionHandler(movieData, "regressionResults-revenue-budgetLog&runtimeSquare",
                      {"revenueNorm": lambda x: x*(colInfo.get("revenueNorm")[2]-colInfo.get("revenueNorm")[1])+colInfo.get("revenueNorm")[1],
                       "revenueLog": lambda x: math.exp(x*(colInfo.get("revenueLog")[2]-colInfo.get("revenueLog")[1])+colInfo.get("revenueLog")[1])},
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
                      {"budgetLog": ["budgetLog"], "runtimeSquare": ["runtimeSquare"]}, useTF = False);

# 1d: 基本模型, 固定使用 rumtime-square, budget-log => revenue
autoRegressionHandler(movieData, "regressionResults-revenue-budgetSquare&runtimeSquare",
                      {"revenueNorm": lambda x: x*(colInfo.get("revenueNorm")[2]-colInfo.get("revenueNorm")[1])+colInfo.get("revenueNorm")[1],
                       "revenueLog": lambda x: math.exp(x*(colInfo.get("revenueLog")[2]-colInfo.get("revenueLog")[1])+colInfo.get("revenueLog")[1])},
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
                      {"budgetSquare": ["budgetSquare"], "runtimeSquare": ["runtimeSquare"]}, useTF = False);


# 2a: 基本模型 => popularity
autoRegressionHandler(movieData, "regressionResults-popularity-basic",
                      {"popularityNorm": lambda x: x*(colInfo.get("popularityNorm")[2]-colInfo.get("popularityNorm")[1])+colInfo.get("popularityNorm")[1]},
                      {"year": ["year"], "budget": ["budget"], "runtime": ["runtime"],
                       "cast": ["Cast_121323_Bess Flowers","Cast_113_Christopher Lee","Cast_4165_John Wayne","Cast_2231_Samuel L. Jackson","Cast_3895_Michael Caine","Cast-Seg-1","Cast-Seg-2","Cast-Seg-3","Cast-Seg-4"],
                       "crew": ["Crew_9062_Cedric Gibbons","Crew_2952_Avy Kaufman","Crew_4350_Edith Head","Crew_102429_Roger Corman","Crew_1259_Ennio Morricone","Crew-Seg-1","Crew-Seg-2","Crew-Seg-3","Crew-Seg-4"],
                       "genre": ["Genre_18_Drama","Genre_35_Comedy","Genre_53_Thriller","Genre_10749_Romance","Genre_28_Action","Genre-Seg-1","Genre-Seg-2"],
                       "production_company": ["Production Company_6194_Warner Bros.","Production Company_8411_Metro-Goldwyn-Mayer (MGM)","Production Company_4_Paramount Pictures","Production Company_306_Twentieth Century Fox Film Corporation","Production Company_33_Universal Pictures"],
                       "month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"],
                       "production_country": ["Production Country_US_United States of America","Production Country_GB_United Kingdom","Production Country_FR_France","Production Country_DE_Germany","Production Country_IT_Italy","Production Country-Seg-1","Production Country-Seg-2","Production Country-Seg-3"]
                       },
                      {"budget": lambda val: val is not None,
                       "runtime": lambda val: val > 0}, useTF = False, oriOutputCol = "popularityNorm");

# 2b: 基本模型, 固定使用 rumtime-square, budget-log => popularity
autoRegressionHandler(movieData, "regressionResults-popularity-runtimeSquare",
                      {"popularityNorm": lambda x: x*(colInfo.get("popularityNorm")[2]-colInfo.get("popularityNorm")[1])+colInfo.get("popularityNorm")[1]},
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
                      {"runtimeSquare": ["runtimeSquare"]}, useTF = False, oriOutputCol = "popularityNorm");

# 2c: 基本模型, 固定使用 rumtime-square, budget-log => popularity
autoRegressionHandler(movieData, "regressionResults-popularity-budgetLog&runtimeSquare",
                      {"popularityNorm": lambda x: x*(colInfo.get("popularityNorm")[2]-colInfo.get("popularityNorm")[1])+colInfo.get("popularityNorm")[1]},
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
                      {"budgetLog": ["budgetLog"], "runtimeSquare": ["runtimeSquare"]}, useTF = False, oriOutputCol = "popularityNorm");

# 2d: 基本模型, 固定使用 rumtime-square, budget-log => popularity
autoRegressionHandler(movieData, "regressionResults-popularity-budgetSquare&runtimeSquare",
                      {"popularityNorm": lambda x: x*(colInfo.get("popularityNorm")[2]-colInfo.get("popularityNorm")[1])+colInfo.get("popularityNorm")[1]},
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
                      {"budgetSquare": ["budgetSquare"], "runtimeSquare": ["runtimeSquare"]}, useTF = False, oriOutputCol = "popularityNorm");


# 3a: 基本模型 => revpop
autoRegressionHandler(movieData, "regressionResults-revPop-basic",
                      {"revPop": lambda x: x},
                      {"year": ["year"], "budget": ["budget"], "runtime": ["runtime"],
                       "cast": ["Cast_121323_Bess Flowers","Cast_113_Christopher Lee","Cast_4165_John Wayne","Cast_2231_Samuel L. Jackson","Cast_3895_Michael Caine","Cast-Seg-1","Cast-Seg-2","Cast-Seg-3","Cast-Seg-4"],
                       "crew": ["Crew_9062_Cedric Gibbons","Crew_2952_Avy Kaufman","Crew_4350_Edith Head","Crew_102429_Roger Corman","Crew_1259_Ennio Morricone","Crew-Seg-1","Crew-Seg-2","Crew-Seg-3","Crew-Seg-4"],
                       "genre": ["Genre_18_Drama","Genre_35_Comedy","Genre_53_Thriller","Genre_10749_Romance","Genre_28_Action","Genre-Seg-1","Genre-Seg-2"],
                       "production_company": ["Production Company_6194_Warner Bros.","Production Company_8411_Metro-Goldwyn-Mayer (MGM)","Production Company_4_Paramount Pictures","Production Company_306_Twentieth Century Fox Film Corporation","Production Company_33_Universal Pictures"],
                       "month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"],
                       "production_country": ["Production Country_US_United States of America","Production Country_GB_United Kingdom","Production Country_FR_France","Production Country_DE_Germany","Production Country_IT_Italy","Production Country-Seg-1","Production Country-Seg-2","Production Country-Seg-3"]
                       },
                      {"budget": lambda val: val is not None,
                       "runtime": lambda val: val > 0}, useTF = False, oriOutputCol = "revPop");
