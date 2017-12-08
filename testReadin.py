import csv;

with open("movies_metadata.csv", newline='', encoding='utf-8-sig') as f:
    reader = csv.reader(f);
    allData = list();
    i = 0;
    for row in reader:
        if (i>0):
            allData.append(row);
        i+=1;
    print([x for x in filter(lambda row: len(row) != len(allData[0]), allData)]);
    print(len(allData[0]));
