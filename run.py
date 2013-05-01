__author__ = 'ltorres'

import pandas
from kmeanspds import *

def run_mouse():
    path = "./mouse.csv"
    df = pandas.read_csv(path, delimiter=" ").ix[:,:2]

    print "head(mouse):"
    print(df.head())

    kmeans = KMeans(3, df)
    outdf = kmeans.run()

    print kmeans

    outpath = "./mouseclusters.csv"
    outdf.to_csv(outpath)

    print "\nWrote clusters to %s" % outpath

if __name__ == '__main__':
    run_mouse()