import csv

lst = [.1]*10
with open('test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['id,target'])
    for i in range(len(lst)):
        writer.writerow([str(i) + ',' + str(lst[i])])
