
import sqlContext.implicits._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature._
import org.apache.spark.ml.Pipeline

// improve performance on MacOSX
//import org.apache.hadoop.fs.RawLocalFileSystem
//RawLocalFileSystem.useStatIfAvailable

import org.nd4j.linalg.factory.Nd4j
val array = Nd4j.create(1)

sc.hadoopConfiguration.set("mapreduce.input.fileinputformat.input.dir.recursive","true")

sc.hadoopConfiguration.get("fs.defaultFS")

// spark.localExecution.enabled

val lfwPath = "/samples/lfw"
//val lfwRDD = sc.binaryFiles(lfwPath)


import org.apache.spark.sql.sources.lfw._
val df = sqlContext.lfw("/samples/lfw")

import org.apache.spark.ml.Pipeline

/*val pipeline = (new Pipeline()
  .setStages(Array(
    //new ImageExtractor().setInputCol("file").setOutputCol("features")  
      new CanovaImageExtractor(28,28).setInputCol("file").setOutputCol("features")  
    )))

val df2 = pipeline.fit(df).transform(df)
df2.show*/
