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
    return jsarray.map(lambda row: JS.Map([([header[i], cell] if not JSONColNames.includes(header[i]) else [header[i], JS.AST.parse(cell)]) for i,cell in enumerate(row)]));

print("Reading Movie Data...");
movieHeader, movieData = readCSV("movies_metadata (lineBreakFixed).csv", True);
print("Reading Credit Data...");
creditHeader, creditData = readCSV("credits.csv", True);

print("Converting Data into objects - Movie...");
movieData = tableToObject(movieData, movieHeader, JS.Array(["genres", "production_companies", "production_countries", "spoken_languages"]));
# this part may be slow because many long strings need to be parsed
print("Converting Data into objects - Credit...");
creditData = tableToObject(creditData, creditHeader, JS.Array(["cast", "crew"]));

# status only released?
print("\nWhat are the values in \"status\" in movie metadata?\n");
print(JS.Set(movieData.map(lambda v: v.get("status"))));
# check blank value records?
print("\nFind out one record with blank status to check with:\n");
print(movieData.filter(lambda v: v.get("status")=="")[0]);

# check parsing results (credit file)
print("\nFind out one record with ast representation to check with:\n");
print(creditData[0].get("cast"));

# find out all movies "Steve Martin" participated in
# can be verified by searching 'name': 'Steve Martin' in Notepad++, then distinct no. of columns in Excel
print("\nFind out all movies which \"Steve Martin\" is involved:\n");
# Step 1: Filter the creditData, which contains cast or crew including Steve Martin in the name attribute, then we collect the id
allMoviesID = creditData.filter(lambda credit: credit.get("cast").some(lambda cast: cast.get("name") == "Steve Martin") or credit.get("crew").some(lambda cast: cast.get("name") == "Steve Martin")).map(lambda v: v.get("id"));
# Step 2: Find out the movie records using the id collected
allMovies = movieData.filter(lambda movie: allMoviesID.includes(movie.get("id")));
print("Movie Count: ", allMoviesID.length);
print(allMovies.map(lambda movie: movie.get("original_title")));

