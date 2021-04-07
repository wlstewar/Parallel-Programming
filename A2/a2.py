import pyspark as ps
import sys
import os


'''

        Connected components in Apache Spark (Map-Reduce model)
        
        Algorithm:  Follows directly from in-class lectures.  We do rounds of
        largeStar (until no change is detected) followed by a round of smallStar
        (following the notes).  To detect changes, the algorithm sums the keys,
        and then continues if the sum is less than the previous sum.  Since we
        always hook to smaller vertices, this sum is guaranteed to decrease
        until no change can be made (for both smallStar and largeStar).  At
        which point we're either done with a largeStar round or done with
        the algorithm.
        
        helper functions:
        
        smallStar - initial mapper for smallStar setup
        largeStar - initial mapper for largeStar setup
        
        ss - smallStar reducer but used w/ flatMap
        ll - largeStar reducer but used w/ flatMap 
        
        key_sum - maps keys into an rdd and computes sum (for detecting changes)
        
        edge - parses lines into k,v pairs

'''

def usage():
        print('''usage:  a2.py <src dir/file> <dest dir/file>''')

def smallStar(p):
        return ((p[1],[p[0]]) if p[0]<p[1] else (p[0],[p[1]]))

def ss(e):
        m = min(e[0],min(e[1]))
        return [(m,v) for v in e[1]+[e[0]]]

def largeStar(p):
        return [(p[0],[p[1]]), (p[1],[p[0]])]

def ll(e):
        m = min(e[0],min(e[1]))
        return [(v,m) for v in e[1]+[e[0]] if e[0] < v]

def key_sum(rdd):
        return rdd.map(lambda x:x[0]).reduce(lambda x,y:x+y)

def edge(line):
        v = [int(x) for x in line.split(' ')]
        return ( min(v[0], v[1]), max(v[0],v[1]))

if __name__ == '__main__':
        if len(sys.argv) < 3:
                usage()
                exit(1)

        # init spark
        sc = ps.SparkContext()

        # create rdd
        E = sc.textFile(sys.argv[1]).map(edge)
        start_sum = float('inf')
        
        # find connected components
        while True:
                while True:
                        start_sum = key_sum(E)
                        E = E.flatMap(largeStar).reduceByKey(lambda x,y:x+y)\
                                .flatMap(ll).distinct()
                        end_sum = key_sum(E)
                        if end_sum >= start_sum:
                                break
                E = E.map(smallStar).reduceByKey(lambda x,y:x+y)\
                        .flatMap(ss).distinct()
                end_sum = key_sum(E)
                if end_sum >= start_sum:
                        break

        # write to file
        E.map(lambda x: (x[1],x[0])).saveAsTextFile(sys.argv[2])
        sc.stop()
