import csv;
import asyncio;
import dexJS as JS;
import sys;

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
    # toAry = jsarray.map(lambda row: JS.Map([([header[i], cell] if not JSONColNames.includes(header[i]) else [header[i], JS.JSON.parse(cell)]) for i,cell in enumerate(row)]));
    return jsarray.map(lambda row: JS.Map([[header[i], cell] for i,cell in enumerate(row)]));

print("Reading Movie Data...");
movieHeader, movieData = readCSV("movies_metadata (lineBreakFixed).csv", True);
print("Reading Credit Data...");
creditHeader, creditData = readCSV("credits.csv", True);

print("Converting Movie Data into objects - Movie...");
movieData = tableToObject(movieData, movieHeader, JS.Array(["genres", "production_companies", "production_countries", "spoken_languages"]));
print("Converting Movie Data into objects - Credit...");
creditData = tableToObject(creditData, creditHeader, JS.Array(["cast", "crew"]));

# status only released?
valueSet = JS.Set();
movieData.forEach(lambda v,i,ary: valueSet.add(v.get("status")));
print(valueSet);
