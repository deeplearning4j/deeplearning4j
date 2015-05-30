
package org.apache.spark.sql.sources.lfw

import org.apache.hadoop.conf.{Configuration => HadoopConfiguration}
import org.apache.hadoop.fs.Path
import org.apache.hadoop.mapreduce._
import org.apache.hadoop.mapreduce.lib.input.{CombineFileInputFormat, CombineFileRecordReader, CombineFileSplit}
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.sources._
import org.apache.spark.sql.sources.canova.{CanovaImageVectorizer, CanovaRecordReaderAdapter}
import org.apache.spark.sql.sources.mapreduce.{PrunedReader, CachedStatus, LabelRecordReader, ColumnarRecordReader}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SQLContext}

/**
 * LFW dataset as a Spark SQL relation.
 */
case class LfwRelation(location: String)(@transient val sqlContext: SQLContext)
  extends BaseRelation
  with PrunedScan with Logging {

  override def schema: StructType = StructType(
    StructField("label", StringType, nullable = false) ::
      StructField("features", new VectorUDT, nullable = false) :: Nil)

  override def buildScan(requiredColumns: Array[String]): RDD[Row] = {

    val sc = sqlContext.sparkContext

    val conf = new HadoopConfiguration(sc.hadoopConfiguration)
    PrunedReader.setRequiredColumns(conf, requiredColumns)
    CanovaImageVectorizer.setWidth(conf, 28)
    CanovaImageVectorizer.setHeight(conf, 28)
    conf.setLong("mapreduce.input.fileinputformat.split.maxsize", 10 * 1024 * 1024)
    conf.setBoolean("mapreduce.input.fileinputformat.input.dir.recursive",true)

    val baseRdd = sc.newAPIHadoopFile[String, Row, LfwInputFormat](
      location, classOf[LfwInputFormat], classOf[String], classOf[Row], conf)

    baseRdd.map { case (key, value) => value }
  }

  override def hashCode(): Int = 41 * (41 + location.hashCode) + schema.hashCode()

  override def equals(other: Any): Boolean = other match {
    case that: LfwRelation =>
      (this.location == that.location) && this.schema.sameType(that.schema)
    case _ => false
  }
}

private class LfwInputFormat
  extends CombineFileInputFormat[String, Row]
  with CachedStatus
{
  override protected def isSplitable(context: JobContext, file: Path): Boolean = false

  override def createRecordReader(split: InputSplit, taContext: TaskAttemptContext)
  : RecordReader[String, Row] = {
    new ColumnarRecordReader(
      ("label", new CombineFileRecordReader[String, Row](
        split.asInstanceOf[CombineFileSplit], taContext, classOf[LabelRecordReader])),
      ("features", new CombineFileRecordReader[String, Row](
        split.asInstanceOf[CombineFileSplit], taContext, classOf[LfwImageRecordReader]))
      )
  }
}

class LfwImageRecordReader(split: CombineFileSplit, context: TaskAttemptContext, index: Integer)
  extends CanovaRecordReaderAdapter(split, context, index)
  with CanovaImageVectorizer {
}






