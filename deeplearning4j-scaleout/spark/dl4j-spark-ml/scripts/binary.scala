
import sqlContext.implicits._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature._
import org.apache.spark.ml.Pipeline
import org.deeplearning4j.spark.sql.sources.lfw._
import org.nd4j.linalg.factory.Nd4j
val array = Nd4j.create(1)

// improve performance on MacOSX
//import org.apache.hadoop.fs.RawLocalFileSystem
//RawLocalFileSystem.useStatIfAvailable

val df = sqlContext.lfw("/samples/lfw")
