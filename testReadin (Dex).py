import csv;
import asyncio;
import dexJS as JS;
import sys;
import re;

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
'''
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
'''

# check if all movies in credit can be found in movie
allCredMovieSet = JS.Set(creditData.map(lambda v: v.get("id")));
allCreditMovies = JS.Array([*allCredMovieSet]);
allMovieMovieSet = JS.Set(movieData.map(lambda v: v.get("id")));
allMovieMovies = JS.Array([*allMovieMovieSet]);
creditOnly = allCreditMovies.filter(lambda v: not allMovieMovieSet.has(v));
movieOnly = allMovieMovies.filter(lambda v: not allCredMovieSet.has(v));
print("\n\nOnly exist in Credits: ", creditOnly.length, "\nOnly exist in Movies: ", movieOnly.length);

# Result shows only one record in movieOnly
print("\n\nThe record only: ");
print(movieData.filter(lambda m: movieOnly.includes(m.get("id"))));

# Value set in Adult
print("\n\nAll Movie Count: ", movieData.length);


# Value set in Adult
print("\n\nValues in Adult: ");
print(JS.Set(movieData.map(lambda v: v.get("adult"))));

# How Many belongs to collections
print("\n\nValues in All Collections: ");
allCol = JS.Map();
movieData.forEach(lambda v,k,self: allCol.set(v.get("belongs_to_collection").get("id"), {"name": v.get("belongs_to_collection").get("name"), "count": ((allCol.get(v.get("belongs_to_collection").get("id")) or {"count": 0}).get("count") + 1)}) if v.get("belongs_to_collection") else True);
print("\nCount: ", allCol.size);
allColCountsAry = JS.Array([*allCol]);
allColCountsAry.sort(lambda a,b: 1 if float(a[1].get("count")) < float(b[1].get("count")) else -1);
print("\nSorted Collections: ", allColCountsAry.slice(0,5));

# Budgets
print("\n\nValues in Budgets: ");
allBudgets = movieData.map(lambda v: float(str(v.get("budget")))).filter(lambda v: v>0);
print("\nMin: ", min(allBudgets));
print("\nMax: ", max(allBudgets));
mean = sum(allBudgets)/allBudgets.length;
print("\nMean: ", mean);
print("\nStandard Deviation: ", (sum(JS.Array([*allBudgets]).map(lambda x: (x-mean)**2))/allBudgets.length)**0.5);

# Values in Genre
print("\n\nValues in All Genre: ");
genreMap = JS.Map();
movieData.forEach(lambda v,k,self: v.get("genres").forEach(lambda g,k2,self2: genreMap.set(g.get("id"), {"name": g.get("name"), "count": ((genreMap.get(g.get("id")) or {"count": 0}).get("count") + 1)})) if v.get("genres") else True);
print("\nCount: ", genreMap.size);
print("\nValues: \n", JS.Array([*genreMap.values()]).map(lambda g: g.get("name")));
allGenAry = JS.Array([*genreMap]);
print("\nSorted Collections: ", allGenAry.sort(lambda a,b: 1 if float(a[1].get("count")) < float(b[1].get("count")) else -1).slice(0,5));

# Original Languages
print("\n\nValues in All Original Languages: ");
allLang = JS.Map();
movieData.forEach(lambda v,k,self: allLang.set(v.get("original_language"), (allLang.get(v.get("original_language")) or 0) + 1) if v.get("original_language") else True);
print("\nCount: ", allLang.size);
allLangCountsAry = JS.Array([*allLang]);
print("\nSorted Lang: ", allLangCountsAry.sort(lambda a,b: 1 if float(a[1]) < float(b[1]) else -1).slice(0,5));

# Popularity
print("\n\nValues in Popularity: ");
allPopularity = movieData.map(lambda v: float(str(v.get("popularity")))).filter(lambda v: v>0);
print("\nMin: ", min(allPopularity));
print("\nMax: ", max(allPopularity));
mean = sum(allPopularity)/allPopularity.length;
print("\nMean: ", mean);
print("\nStandard Deviation: ", (sum(JS.Array([*allPopularity]).map(lambda x: (x-mean)**2))/allPopularity.length)**0.5);

# Values in Production Companies
print("\n\nValues in All Production Companies: ");
prodComMap = JS.Map();
movieData.forEach(lambda v,k,self: v.get("production_companies").forEach(lambda g,k2,self2: prodComMap.set(g.get("id"), {"name": g.get("name"), "count": ((prodComMap.get(g.get("id")) or {"count": 0}).get("count") + 1)})) if v.get("production_companies") else True);
print("\nCount: ", prodComMap.size);
allProdCompAry = JS.Array([*prodComMap]);
print("\nSorted Production Companies: ", allProdCompAry.sort(lambda a,b: 1 if float(a[1].get("count")) < float(b[1].get("count")) else -1).slice(0,5));

# Values in Production Countries
print("\n\nValues in All Production Countries: ");
prodConMap = JS.Map();
movieData.forEach(lambda v,k,self: v.get("production_countries").forEach(lambda g,k2,self2: prodConMap.set(g.get("iso_3166_1"), {"name": g.get("name"), "count": ((prodConMap.get(g.get("iso_3166_1")) or {"count": 0}).get("count") + 1)})) if v.get("production_countries") else True);
print("\nCount: ", prodConMap.size);
allProdContAry = JS.Array([*prodConMap]);
print("\nSorted Production Countries: ", allProdContAry.sort(lambda a,b: 1 if float(a[1].get("count")) < float(b[1].get("count")) else -1).slice(0,5));

# Values in Release Date
print("\n\nValues in All Release Dates: ");
allDates = movieData.map(lambda v: v.get("release_date")).map(lambda v: JS.Map([["year", int(str(v.slice(0,4)))], ["month", int(str(v.slice(5,7)))], ["date", int(str(v.slice(8,10)))]]) if v.length else JS.null);
allYearMap = JS.Map();
allDates.forEach(lambda v,k,self: allYearMap.set(v.get("year"), (allYearMap.get(v.get("year")) or 0) + 1) if v else True);
print("\nMin Year: ", min(*allYearMap.keys()));
print("\nMax Year: ", max(*allYearMap.keys()));
allYearAry = JS.Array([*allYearMap]);
print("\nSorted Years: ", allYearAry.sort(lambda a,b: 1 if float(a[1]) < float(b[1]) else -1).slice(0,5));
allMonthsMap = JS.Map();
allDates.forEach(lambda v,k,self: allMonthsMap.set(v.get("month"), (allMonthsMap.get(v.get("month")) or 0) + 1) if v else True);
allMonthsAry = JS.Array([*allMonthsMap]);
print("\nSorted Months: ", allMonthsAry.sort(lambda a,b: 1 if float(a[1]) < float(b[1]) else -1).slice(0,5));

# Revenue
print("\n\nValues in Revenue: ");
allRevenue = movieData.map(lambda v: float(str(v.get("revenue")))).filter(lambda v: v>0);
print("\nMin: ", min(allRevenue));
print("\nMax: ", max(allRevenue));
mean = sum(allRevenue)/allRevenue.length;
print("\nMean: ", mean);
print("\nStandard Deviation: ", (sum(JS.Array([*allRevenue]).map(lambda x: (x-mean)**2))/allRevenue.length)**0.5);

# Run Time
print("\n\nValues in Runtime: ");
allRuntime = movieData.map(lambda v: float(str(v.get("runtime")) if v.get("runtime") else 0)).filter(lambda v: v>0);
print("\nMin: ", min(allRuntime));
print("\nMax: ", max(allRuntime));
mean = sum(allRuntime)/allRuntime.length;
print("\nMean: ", mean);
print("\nStandard Deviation: ", (sum(JS.Array([*allRuntime]).map(lambda x: (x-mean)**2))/allRuntime.length)**0.5);
suspiciousMovies = movieData.filter(lambda v: float(str(v.get("runtime")))>=240 if v.get("runtime") else False).map(lambda v: JS.Array([v.get("original_title"), float(str(v.get("runtime")))]));
print("\nPartial Suspiciously Long Movies Count: ", suspiciousMovies.length);
print("\nPartial Suspiciously Long Movies: ", suspiciousMovies.sort(lambda a,b: 1 if float(a[1]) < float(b[1]) else -1).slice(0,5));
print("\nTop Suspicious Records: ", movieData.filter(lambda v: v.get("original_title") == 'Centennial')[0])

# Spoken Languages
print("\n\nValues in All Spoken Languages: ");
allSpkLang = JS.Map();
movieData.forEach(lambda v,k,self: v.get("spoken_languages").forEach(lambda g,k2,self2: allSpkLang.set(g.get("iso_639_1"), {"name": g.get("name"), "count": ((allSpkLang.get(g.get("iso_639_1")) or {"count": 0}).get("count") + 1)})) if v.get("spoken_languages") else True);
print("\nCount: ", allSpkLang.size);
allSpkLangCountsAry = JS.Array([*allSpkLang]);
print("\nSorted Spoken Languages: ", allSpkLangCountsAry.sort(lambda a,b: 1 if float(a[1].get("count")) < float(b[1].get("count")) else -1).slice(0,5));

# Status
print("\n\nValues in All Status: ");
allStatus = JS.Map();
movieData.forEach(lambda v,k,self: allStatus.set(v.get("status"), (allStatus.get(v.get("status")) or 0) + 1) if v.get("status") else True);
print("\nCount: ", allStatus.size);
allStatusAry = JS.Array([*allStatus]);
print("\nSorted Status: ", allStatusAry.sort(lambda a,b: 1 if float(a[1]) < float(b[1]) else -1));

#Overview
print("\n\nValues in Overview: ");
allVocab = JS.Map();
movieData.forEach(lambda v,k,self: v.get("overview").split(re.compile("\s")).forEach(lambda g,k2,self2: allVocab.set(g.toLowerCase(), (allVocab.get(g.toLowerCase()) or 0) + 1)) if v.get("spoken_languages") else True);
allVocabAry = JS.Array([*allVocab]).filter(lambda v: v[0].trim());
print("\nCount: ", allVocabAry.length);
hotwords = allVocabAry.sort(lambda a,b: 1 if float(a[1]) < float(b[1]) else -1)
print("\nSorted Vocab: ", hotwords.slice(0,20));
print("\nMore Sorted Vocab: ", hotwords.slice(0,100).map(lambda v: v[0]).join(", "));

#People
print("\n\nPeople: ");
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
print("\nPeople Count: ", allPeopleBio.size);
allCastAry = JS.Array([*allPeopleCastCount]);
allCastAry.sort(lambda a,b: 1 if float(a[1]) < float(b[1]) else -1);
print("\nSorted Cast: ", allCastAry.slice(0,10).map(lambda p: [allPeopleBio.get(p[0])["name"], p[1]]));
allCrewAry = JS.Array([*allPeopleCrewCount]);
allCrewAry.sort(lambda a,b: 1 if float(a[1]) < float(b[1]) else -1);
print("\nSorted Crew: ", allCrewAry.slice(0,10).map(lambda p: [allPeopleBio.get(p[0])["name"], p[1]]));
allCredAry = allCastAry.map(lambda p: [p[0],p[1]+allPeopleCrewCount.get(p[0])]);
allCredAry.sort(lambda a,b: 1 if float(a[1]) < float(b[1]) else -1);
print("\nSorted Credit: ", allCredAry.slice(0,10).map(lambda p: [allPeopleBio.get(p[0])["name"], p[1]]));
