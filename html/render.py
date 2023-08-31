from jinja2 import Template
import csv
import os

root = os.getcwd()
class Fruit:
    def __init__(self, index, name, min, max):
        self.index = index
        self.name = name
        self.min = min
        self.max = max

def html_to_string(file_address):
    result = ""
    with open(file_address, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line.replace("\n", '')
            result+=line
    return result

fruit_lst = []
with open(root+'\database.csv', 'r') as db:
    dbr = csv.reader(db, delimiter=' ')
    i = 1
    for row in dbr:
        if len(row):
            cont = row[0].split(',')
            print(cont)
            fruit_lst.append(Fruit(i, cont[0], cont[1], cont[2]))
            i += 1

hstr = html_to_string('html\sample.html')
page = Template(hstr).render(classes = fruit_lst)
with open("html\index.html", "w", encoding="utf-8") as f:
    f.write(page)