import sys
from pyspark.sql import SparkSession, functions, types
 
# ADD the following lines before doing anything 
# SET JAVA_HOME=C:\Program Files\Android\Android Studio\jre
# SET HADOOP_HOME=C:\winutils
 
spark = SparkSession.builder.appName('reddit relative scores').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
 
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+
 
comments_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('controversiality', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('id', types.StringType()),
    types.StructField('link_id', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('parent_id', types.StringType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('score_hidden', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('ups', types.LongType()),
    #types.StructField('year', types.IntegerType()),
    #types.StructField('month', types.IntegerType()),
])
 
 
def main(in_directory, out_directory):
    comments = spark.read.json(in_directory, schema=comments_schema)
    
    # TODO
    avg = comments.groupby(comments['subreddit']).avg("score")
    avg = avg.cache()
    
    avg = avg.filter(
        avg["avg(score)"] > 0.0
    )
    
    avg = avg.join(comments, 'subreddit', 'inner')
    
    avg = avg.withColumn('relative_score', avg['score']/avg['avg(score)'])
    
    
    relative_score = avg.groupby('subreddit').agg(functions.max('relative_score').alias('relative_score'))
    
    
    #relative_score['sub'] = relative_score['subreddit']
    #relative_score.drop('subreddit')
    relative_score = relative_score.select(
    	relative_score["subreddit"].alias("sub"),
    	relative_score["relative_score"].alias("relative_score")
    	)
    bAuthor = relative_score.join(avg, on = "relative_score")
    
   
    bAuthor = bAuthor.select(
    	"subreddit",
    	"author",
    	"relative_score",    	
    	)
    	
    bAuthor.show()
    
if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
