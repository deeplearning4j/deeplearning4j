package org.deeplearning4j.spark.sql.sources.mnist


import java.net.URI
import java.nio.file.Paths

import org.apache.hadoop.fs.Path
import org.apache.spark.{SparkContext, Partition, TaskContext, Logging}
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
 * @author Kai Sasaki, Eron Wright
 */
case class MnistRelation(
                          imagesPath: Path,
                          labelsPath: Path,
                          recordsPerPartition: Int,
                          maxRecords: Option[Int])
    (@transient val sqlContext: SQLContext) extends BaseRelation
    with PrunedScan with Logging  {

  override def schema: StructType = StructType(
    StructField("label", DoubleType, nullable = false) ::
      StructField("features", VectorUDT(), nullable = false) :: Nil)

  override def buildScan(requiredColumns: Array[String]): RDD[Row] = {
    val sc = sqlContext.sparkContext

    new MnistRDD(sc, imagesPath.toUri, labelsPath.toUri, requiredColumns, recordsPerPartition, maxRecords)
  }
}

private class MnistPartition(override val index: Int, val startIndex: Int, val endIndex: Int)
  extends Partition with Serializable {
}

private class MnistRDD(
    sc: SparkContext,
    private val imagesPath: URI,
    private val labelsPath: URI,
    private val requiredColumns: Array[String],
    recordsPerPartition: Int = 1000,
    maxRecords: Option[Int] = None
  )
  extends RDD[Row](sc, Nil) {

  override val partitioner = None

  override def getPartitions: Array[Partition] = {
    val manager = open()
    try {
      val numLabels = manager.getLabels.getCount
      val numRecords = Math.min(numLabels, maxRecords.getOrElse { numLabels })
      val numPartitions = Math.ceil(numRecords / (recordsPerPartition: Double)).asInstanceOf[Int]
      val array = new Array[Partition](numPartitions)
      for (i <- 0 until numPartitions) {
        array(i) = new MnistPartition(
          i,
          i * recordsPerPartition,
          Math.min((i + 1) * recordsPerPartition - 1, numRecords - 1))
      }
      array
    }
    finally {
      manager.close()
    }
  }

  override def compute(split: Partition, context: TaskContext): Iterator[Row] = {
    val manager = open()
    context.addTaskCompletionListener((_) => manager.close())

    val msplit = split.asInstanceOf[MnistPartition]

    val cols = {
      val req = requiredColumns.toSet
      Seq("label", "features") flatMap {
        case "label" if req("label") => Seq(
          () => manager.readLabel().toDouble)
        case "features" if req("features") => Seq(
          () => Vectors.dense(ArrayUtil.flatten(manager.readImage()).map(_.toDouble)))
        case _ => Seq.empty
      }
    }

    manager.setCurrent(msplit.startIndex)

    (msplit.startIndex to msplit.endIndex).map((_) => {
      Row.fromSeq(cols.map(_()))
    }).iterator
  }

  private def open(): MnistManager = {
    // CAVEAT: MnistManager supports only local files at this time.
    val imagesFile = Paths.get(imagesPath).toFile.getAbsolutePath
    val labelsFile = Paths.get(labelsPath).toFile.getAbsolutePath
    new MnistManager(imagesFile, labelsFile)
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

  private def checkRecordsPerPartition(parameters: Map[String, String]): Int = {
    parameters.getOrElse(RecordsPerPartition, 1000) match {
      case r: String => Integer.parseInt(r)
      case r: Int => r
    }
  }

  private def checkMaxRecords(parameters: Map[String, String]): Option[Int] = {
    parameters.getOrElse(MaxRecords, None) match {

      case r: String => Option(Integer.parseInt(r))
      case None => None
    }
  }

  override def createRelation(sqlContext: SQLContext,
      parameters: Map[String, String]) = {
    val imagesPath = new Path(checkImagesFilePath(parameters))
    val labelsPath = new Path(checkLabelsFilePath(parameters))
    val recordsPerPartition = checkRecordsPerPartition(parameters)
    val maxRecords = checkMaxRecords(parameters)
    new MnistRelation(imagesPath, labelsPath, recordsPerPartition, maxRecords)(sqlContext)
  }
}

object DefaultSource {
  val ImagesPath = "imagesPath"
  val LabelsPath = "labelsPath"
  val RecordsPerPartition = "recordsPerPartition"
  val MaxRecords = "maxRecords"
}