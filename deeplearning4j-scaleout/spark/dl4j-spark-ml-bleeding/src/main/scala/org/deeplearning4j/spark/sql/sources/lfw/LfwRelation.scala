/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.spark.sql.sources.lfw

import org.apache.hadoop.conf.{Configuration => HadoopConfiguration}
import org.apache.hadoop.fs.Path
import org.apache.hadoop.mapreduce._
import org.apache.hadoop.mapreduce.lib.input.{CombineFileInputFormat, CombineFileRecordReader, CombineFileSplit}
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SQLContext}
import org.deeplearning4j.spark.sql.sources.canova.{CanovaRecordReaderAdapter, CanovaImageVectorizer}
import org.deeplearning4j.spark.sql.sources.mapreduce.{PrunedReader, ColumnarRecordReader, LabelRecordReader, CachedStatus}
import org.deeplearning4j.spark.sql.types.VectorUDT

/**
 * LFW dataset as a Spark SQL relation.
 *
 * @author Eron Wright
 */
case class LfwRelation(location: String)(@transient val sqlContext: SQLContext)
  extends BaseRelation
  with PrunedScan with Logging {

  override def schema: StructType = StructType(
    StructField("label", StringType, nullable = false) ::
      StructField("features", VectorUDT(), nullable = false) :: Nil)

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
      (this.location == that.location) && this.schema.equals(that.schema)
    case _ => false
  }
}

/**
 * LFW dataset provider.
 */
class DefaultSource extends RelationProvider {
  private def checkPath(parameters: Map[String, String]): String = {
    parameters.getOrElse("path", sys.error("'path' must be specified for LFW data."))
  }

  override def createRelation(sqlContext: SQLContext, parameters: Map[String, String]) = {
    val path = checkPath(parameters)
    new LfwRelation(path)(sqlContext)
  }
}

/**
 * LFW input format that produces a Row with 'label' and 'features' columns.
 */
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

private class LfwImageRecordReader(split: CombineFileSplit, context: TaskAttemptContext, index: Integer)
  extends CanovaRecordReaderAdapter(split, context, index)
  with CanovaImageVectorizer {
}






