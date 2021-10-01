import sys
from pyspark.sql import SparkSession, functions, types
import string, re
 
# ADD the following lines before doing anything 
# SET JAVA_HOME=C:\Program Files\Android\Android Studio\jre
# SET HADOOP_HOME=C:\winutils
 
spark = SparkSession.builder.appName('reddit relative scores').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
 
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+
 
 
wordbreak = r'[%s\s]+' % (re.escape(string.punctuation),)

def extract(text):
	data = text.select(functions.split(functions.lower(functions.col('value')), wordbreak).alias("data"))
	data = data.select(functions.explode(functions.col('data')))
	return data


def main(in_directory, out_directory):
	text = spark.read.text(in_directory)
	data = extract(text)
	
	words = data.groupBy("col").count().alias("count").cache()
	_sorted = words.filter(
		words['col'] != ""
		).sort(functions.desc("count")).cache()
		
	sortSelect = _sorted.select(
		_sorted['col'].alias('words'), "count")
	
	sortSelect.write.csv(out_directory, mode = 'overwrite')
		


if __name__=='__main__':
	in_directory = sys.argv[1]
	out_directory = sys.argv[2]
	main(in_directory, out_directory)
