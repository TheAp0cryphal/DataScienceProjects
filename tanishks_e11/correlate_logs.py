
import sys
from pyspark.sql import SparkSession, functions, types, Row
import re

spark = SparkSession.builder.appName('correlate logs').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

line_re = re.compile(r"^(\S+) - - \[\S+ [+-]\d+\] \"[A-Z]+ \S+ HTTP/\d\.\d\" \d+ (\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred. Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        return Row(HostName = m[1], bytes=int(m[2]))
    else:
        return None


def not_none(row):
    """
    Is this None? Hint: .filter() with it.
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    # TODO: return an RDD of Row() objects
    rows = log_lines.map(line_to_row).filter(not_none)
    return rows


def main(in_directory):
    logs = spark.createDataFrame(create_row_rdd(in_directory))

    c = logs.groupBy('HostName').count()
    bytes = logs.groupBy('HostName').sum('bytes')
    g = bytes.join(c, on = "HostName")
    g.show()
    group = g.select(
    	(g["count"]/g["count"]).alias("n"),
    	g["count"].alias("numRequests"),
    	(g["count"]*g["count"]).alias("x^2"),
    	g["sum(bytes)"].alias("sum_of_bytes"),
    	(g["sum(bytes)"] * g["sum(bytes)"]).alias("y^2"),
    	(g["count"] * g["sum(bytes)"] ).alias("x*y")
    	)
    summation = group.groupBy().sum()
    pandas = summation.toPandas()
    pandas = pandas.astype(float)

    abc = ((pandas['sum(n)'] * pandas['sum(x*y)']) - (pandas['sum(numRequests)'] * pandas['sum(sum_of_bytes)']))/((((pandas['sum(n)'] * pandas['sum(x^2)']) - (pandas['sum(numRequests)'] * pandas['sum(numRequests)']))**(0.5))*(((pandas['sum(n)'] * pandas['sum(y^2)']) - (pandas['sum(sum_of_bytes)']*pandas['sum(sum_of_bytes)']))**(0.5)))
    r = abc[0]
    print("r = %g\nr^2 = %g" % (r, r**2))


if __name__=='__main__':
    in_directory = sys.argv[1]
    main(in_directory)
    
