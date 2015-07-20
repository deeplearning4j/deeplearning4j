package org.deeplearning4j.spark.sql.sources.mnist


import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.sources.{PrunedScan, BaseRelation, RelationProvider}
import org.deeplearning4j.datasets.mnist.MnistManager
import org.deeplearning4j.spark.sql.types.VectorUDT
import org.nd4j.linalg.util.ArrayUtil

/**
 * Mnist dataset as a Spark SQL relation.
 *
 * @author Kai Sasaki
 */
case class MnistRelation(imagesFile: String, labelsFile: String, numExamples: Int)
    (@transient val sqlContext: SQLContext) extends BaseRelation
    with PrunedScan with Logging  {

  require(numExamples > 0, "Reading images should not be empty")


  val manager = new MnistManager(imagesFile, labelsFile)

  override def schema: StructType = StructType(
    StructField("label", DoubleType, nullable = false) ::
      StructField("features", VectorUDT(), nullable = false) :: Nil)

  override def buildScan(requiredColumns: Array[String]): RDD[Row] = {
    val sc = sqlContext.sparkContext

    val cols = {
      val req = requiredColumns.toSet
      schema.fieldNames flatMap {
        case "label" if req("label") => Seq((pt:Int) =>
          manager.readLabel().toDouble)
        case "features" if req("features") => Seq((pt:Int) =>
          Vectors.dense(ArrayUtil.flatten(manager.readImage()).map(_.toDouble)))
        case _ => Seq.empty
      } //: Seq[Int => Seq[Any]]
    }

    sc.makeRDD((0 until numExamples).map(pt => {
      Row.fromSeq(cols.map(_(pt)))
      //Row.fromSeq(cols.map(_(pt)).reduceOption(_ ++ _).getOrElse(Seq.empty))
    }))
  }
}


/**
 * Mnist dataset provider.
 */
class DefaultSource extends RelationProvider {
  private def checkImagesFilePath(parameters: Map[String, String]): String = {
    parameters.getOrElse("images_file",
      sys.error("'image_file' must be specified for mnist data"))
  }

  private def checkLabelsFilePath(parameters: Map[String, String]): String = {
    parameters.getOrElse("labels_file",
      sys.error("'labels_file' must be specified for mnist labels"))
  }

  private def checkNumExamples(parameters: Map[String, String]): Int = {
    parameters.getOrElse("num_examples", "1000").toInt
  }

  override def createRelation(sqlContext: SQLContext,
      parameters: Map[String, String]) = {
    val imagesFile = checkImagesFilePath(parameters)
    val labelsFile = checkLabelsFilePath(parameters)
    val numExamples = checkNumExamples(parameters)
    new MnistRelation(imagesFile, labelsFile, numExamples)(sqlContext)
  }
}
