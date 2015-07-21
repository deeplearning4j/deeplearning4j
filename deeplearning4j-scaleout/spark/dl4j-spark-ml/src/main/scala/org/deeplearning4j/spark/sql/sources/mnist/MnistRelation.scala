package org.deeplearning4j.spark.sql.sources.mnist


import java.nio.file.Paths

import org.apache.hadoop.fs.Path
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
case class MnistRelation(imagesPath: Path, labelsPath: Path)
    (@transient val sqlContext: SQLContext) extends BaseRelation
    with PrunedScan with Logging  {

  override def schema: StructType = StructType(
    StructField("label", DoubleType, nullable = false) ::
      StructField("features", VectorUDT(), nullable = false) :: Nil)

  override def buildScan(requiredColumns: Array[String]): RDD[Row] = {
    val sc = sqlContext.sparkContext

    val manager = {
      // CAVEAT: MnistManager supports only local files at this time.
      val imagesFile = Paths.get(imagesPath.toUri).toFile.getAbsolutePath
      val labelsFile = Paths.get(labelsPath.toUri).toFile.getAbsolutePath
      new MnistManager(imagesFile, labelsFile)
    }
    try {
      val cols = {
        val req = requiredColumns.toSet
        schema.fieldNames flatMap {
          case "label" if req("label") => Seq(
            (pt:Int) => manager.readLabel().toDouble)
          case "features" if req("features") => Seq(
            (pt:Int) => Vectors.dense(ArrayUtil.flatten(manager.readImage()).map(_.toDouble)))
          case _ => Seq.empty
        }
      }

      val numExamples = manager.getLabels.getCount

      sc.makeRDD((0 until numExamples).map(pt => {
        Row.fromSeq(cols.map(_(pt)))
      }))
    }
    finally {
      manager.close()
    }
  }
}


/**
 * Mnist dataset provider.
 */
class DefaultSource extends RelationProvider {
  import DefaultSource._

  private def checkImagesFilePath(parameters: Map[String, String]): String = {
    parameters.getOrElse(ImagesPath,
      sys.error("'imagesPath' must be specified for mnist data"))
  }

  private def checkLabelsFilePath(parameters: Map[String, String]): String = {
    parameters.getOrElse(LabelsPath,
      sys.error("'labelsPath' must be specified for mnist labels"))
  }

  override def createRelation(sqlContext: SQLContext,
      parameters: Map[String, String]) = {
    val imagesPath = new Path(checkImagesFilePath(parameters))
    val labelsPath = new Path(checkLabelsFilePath(parameters))
    new MnistRelation(imagesPath, labelsPath)(sqlContext)
  }
}

object DefaultSource {
  val ImagesPath = "imagesPath"
  val LabelsPath = "labelsPath"
}