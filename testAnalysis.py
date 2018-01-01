import csv;
import asyncio;
import dexJS as JS;
import sys;
import re;
import numpy as np;
import math;
import ast;
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
def firstTimeReadIn():
    

    '''
    preX = movieData.map(lambda row: JS.Array([float(row.get("budget")), float(row.get("revenue"))]));
    preX = preX.filter(lambda row: row[0]>=1000 and row[1]>=1000);
    preX = np.array(preX);
    np.random.shuffle(preX);
    trainProp = round(len(preX) * .75);
    trainX = preX[0:trainProp];
    testX = preX[trainProp:len(preX)];

    kmeans = KMeans(n_clusters=12, random_state=0).fit(trainX);
    testPrediction = kmeans.predict(testX);
    with open('kmeansTest.csv', 'w', newline='') as csvfile:
        toWriter = csv.writer(csvfile);
        toWriter.writerow(["budget", "revenue"]);
        JS.Array(trainX).forEach(lambda row, i, ary: toWriter.writerow(row));
    '''
    
    creditIDs = JS.Set(creditData.map(lambda credit: credit.get("id")));
    newMovie = movieData.filter(lambda row: float(row.get("budget"))>=1000 and float(row.get("revenue"))>=1000 and row.get("status") == "Released" and creditIDs.has(row.get("id")));
    '''
    def appendPeopleInfo(movie, i, ary, col):
        movie.set(col, creditData.find(lambda credit: credit.get("id") == movie.get("id")).get(col).map(lambda credit: credit.get("id")));
        if (i%100 == 0):
            print("Processing "+col+ ": " + str(math.floor(i/ary.length*100)) + "%");

    print("Append shortened data...\n\n");
    newMovie.forEach(lambda movie, i, ary: appendPeopleInfo(movie, i, ary, "cast"));
    newMovie.forEach(lambda movie, i, ary: appendPeopleInfo(movie, i, ary, "crew"));
    '''
    
    print("Preparing Temp data...\n\n");
    headerPre = JS.Array([*newMovie[0].keys()]);
    header = headerPre.map(lambda k: k if JS.String(k).indexOf(",") == -1 else ("\"" + k + "\""));
    progress = 5; totalL = newMovie.length; dumpable = JS.Array(["Map", "Set", "Array"]);
    with open('tempData.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
        toWriter = csv.writer(csvfile);
        toWriter.writerow(header);
        for i,row in enumerate(newMovie):
            toWriter.writerow([(repr(row.get(h)) if dumpable.includes(JS.classof(row.get(h))) else str(row.get(h))) for h in headerPre]);
            if (math.floor(i/totalL*100)>=progress):
                print("Output - " + str(progress) + "%");
                progress+=5;
    return newMovie;

def oldDataReadIn():
    _, newMovie = readCSV("tempData.csv", True);
    movieData = tableToObject(movieData, movieHeader, JS.Array(["belongs_to_collection", "genres", "production_companies", "production_countries", "spoken_languages"]));

newMovie = firstTimeReadIn();


print("Data Collected");

#People
print("\n\nPeople Info Preparing: ");
allPeopleBio = JS.Map();
allPeopleCastCount = JS.Map();
allPeopleCrewCount = JS.Map();
def pushCreditInfo(v,i,ary):
    def loopCast(v2,i2,ary):
        peopleID = v2.get("id");
        if (not allPeopleBio.has(peopleID)):
            allPeopleBio.set(peopleID, {"name": v2.get("name"), "gender": v2.get("gender")});
            allPeopleCastCount.set(peopleID, 1);
            allPeopleCrewCount.set(peopleID, 0);
        else:
            allPeopleCastCount.set(peopleID, allPeopleCastCount.get(peopleID) + 1);
    def loopCrew(v2,i2,ary):
        peopleID = v2.get("id");
        if (not allPeopleBio.has(peopleID)):
            allPeopleBio.set(peopleID, {"name": v2.get("name"), "gender": v2.get("gender")});
            allPeopleCastCount.set(peopleID, 0);
            allPeopleCrewCount.set(peopleID, 1);
        else:
            allPeopleCrewCount.set(peopleID, allPeopleCrewCount.get(peopleID) + 1);
        
    allCast = v.get("cast");
    if (allCast):
        allCast.forEach(loopCast);
    allCrew = v.get("crew");
    if (allCrew):
        allCrew.forEach(loopCrew);
creditData.forEach(pushCreditInfo);


print("People Information Prepared.\n\n");


class freqCatList:
    def __init__(self, name, idNameCountMap, counterOrExist = 0, topK = 0, bottomK = 0, segment = 4):
        print("Creating Freq Cat List - "+ name);
        # Convert Categorical Data into Freq-Categorical Columns
        # 1. Sort and Count how many each value present
        allList = JS.Array([*idNameCountMap]).sort(lambda a,b: 1 if a[1]["count"]<b[1]["count"] else -1);
        self.name = name;
        self.segment = segment;
        self.topKList = JS.Map(allList.slice(0, topK).map(lambda a: [a[0], name + "_"+ str(a[0]) + "_" +a[1]["name"]]));
        self.counterOrExist = counterOrExist;
        self.bottomKList = JS.Map(allList.slice(allList.length - bottomK, allList.length).map(lambda a: [a[0], name + "_"+ str(a[0]) + "_" +a[1]["name"]]));
        allList = allList.slice(topK, allList.length - bottomK);
        maxCount = allList[0][1]["count"];
        minCount = allList[allList.length - 1][1]["count"];
        self.remList = JS.Map(allList.map(lambda a: [a[0], math.floor((maxCount - a[1]["count"])/(maxCount - minCount)*segment) if a[1]["count"] != minCount else (segment-1)]));
        self.remHeader = JS.Map([(i, (name + "-Seg-" + str(i+1))) for i in range(segment)])
        print("Created Freq Cat List - "+ name);
    def createColumns(self, jsArray, getValFtn):
        name = self.name; totalLength = jsArray.length;
        print("Creating New Columns - "+ name);
        nextProcess = 5;
        for idx,row in enumerate(jsArray):
            valList = JS.Array(getValFtn(row));
            # 1. Check if the topKList and bottomKList are included in this item
            self.topKList.forEach(lambda v,k,m: row.set(v, (1 if valList.includes(k) else 0)));
            self.bottomKList.forEach(lambda v,k,m: row.set(v, (1 if valList.includes(k) else 0)));

            # 2. The remaining values will check with 
            tmpCounter = JS.Array([0 for i in range(self.segment)]);
            valList = valList.filter(lambda v: not(self.topKList.has(v) or self.bottomKList.has(v)));
            if self.counterOrExist == 0:
                for v in valList:
                    if self.remList.get(v) is None:
                        print(v);
                    tmpCounter[self.remList.get(v)] = 1;
            elif self.counterOrExist == 1:
                for v in valList:
                    tmpCounter[self.remList.get(v)] = tmpCounter[self.remList.get(v)] + 1;
            for i,rem in enumerate(tmpCounter):
                row.set(self.remHeader.get(i), rem);

            # Log Progress
            process = math.floor(idx/totalLength*100);
            if (process >= nextProcess):
                print("Finished: " + str(nextProcess) + "%...");
                nextProcess = nextProcess + 5;
        print("Created New Columns - "+ name);
        return jsArray;


'''castFreqCount = freqCatList("Cast", JS.Map(JS.Array([*allPeopleCastCount]).map(lambda row: [row[0], {"name": allPeopleBio.get(row[0])["name"], "count": row[1]}])), topK = 5, segment = 4) ;
crewFreqCount = freqCatList("Crew", JS.Map(JS.Array([*allPeopleCrewCount]).map(lambda row: [row[0], {"name": allPeopleBio.get(row[0])["name"], "count": row[1]}])), topK = 5, segment = 4) ;

castFreqCount.createColumns(newMovie, lambda movie: movie.get("cast"));
'''

def appendToInfo(v,k,self, countMap, columnName):
    countMap.set(v.get(columnName), {"name": v.get(columnName), "count": ((countMap.get(v.get(columnName)) or {"count": 0}).get("count") + 1)}) if v.get(columnName) else True;

#Language
print("\n\nOriginal Language Info Preparing: ");
allLang = JS.Map();
movieData.forEach(lambda v,k,self: appendToInfo(v,k,self, allLang, "original_language"));

langFreqCount  = freqCatList("Original Language", allLang, topK = 2, segment = 3);
langFreqCount.createColumns(newMovie, lambda movie: [movie.get("original_language")]);

def appendToInfo(v,k,self, countMap, columnName, indexName):
    v.get(columnName).forEach(lambda g,k2,self2: countMap.set(g.get(indexName), {"name": g.get("name"), "count": ((countMap.get(g.get(indexName)) or {"count": 0}).get("count") + 1)}) if v.get(columnName) else True);

#Genres
print("\n\nGenre Info Preparing: ");
allGenre = JS.Map();
movieData.forEach(lambda v,k,self: appendToInfo(v,k,self,allGenre,"genres", "id") );


langGenreCount  = freqCatList("Genre", allGenre, topK = 5, segment = 2);
langGenreCount.createColumns(newMovie, lambda movie: movie.get("genres").map(lambda row: row.get("id")));

#production_companies
print("\n\nproduction_companies Info Preparing: ");
allProdCom = JS.Map();
movieData.forEach(lambda v,k,self: appendToInfo(v,k,self,allProdCom,"production_companies", "id") );

langProdComCount  = freqCatList("Production Company", allProdCom, topK = 5, segment = 4);
langProdComCount.createColumns(newMovie, lambda movie: movie.get("production_companies").map(lambda row: row.get("id")));

#production_countries
print("\n\nproduction_countries Info Preparing: ");
allProdCon = JS.Map();
movieData.forEach(lambda v,k,self: appendToInfo(v,k,self,allProdCon,"production_countries", "iso_3166_1") );

langProdConCount  = freqCatList("Production Country", allProdCon, topK = 5, segment = 3);
langProdConCount.createColumns(newMovie, lambda movie: movie.get("production_countries").map(lambda row: row.get("iso_3166_1")));

#spoken_languages
print("\n\nspoken_languages Info Preparing: ");
allSpkLang = JS.Map();
movieData.forEach(lambda v,k,self: appendToInfo(v,k,self,allSpkLang,"spoken_languages", "iso_639_1") );

langSpkLangCount  = freqCatList("Spoken Language", allSpkLang, topK = 1, segment = 3);
langSpkLangCount.createColumns(newMovie, lambda movie: movie.get("spoken_languages").map(lambda row: row.get("iso_639_1")));                  




print("\n\nData 2 Preparing: ");
noPrintCol = JS.Array(["imdb_id", "overview", "poster_path", "video", "tagline", "belongs_to_collection", "genres", "production_companies", "production_countries", "spoken_languages", "cast", "crew"]);
headerPre = JS.Array([*newMovie[0].keys()]).filter(lambda k: not noPrintCol.includes(k));
header = headerPre.map(lambda k: k if JS.String(k).indexOf(",") == -1 else ("\"" + k + "\""));
progress = 5; totalL = newMovie.length; dumpable = JS.Array(["Map", "Set", "Array"]);
with open('tempData2.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
        toWriter = csv.writer(csvfile);
        toWriter.writerow(header);
        for i,row in enumerate(newMovie):
            toWriter.writerow([(repr(row.get(h)) if dumpable.includes(JS.classof(row.get(h))) else str(row.get(h))) for h in headerPre]);
            if (math.floor(i/totalL*100)>=progress):
                print("Output - " + str(progress) + "%");
                progress+=5;

