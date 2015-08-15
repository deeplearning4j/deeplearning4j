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

package org.deeplearning4j.spark.sql.sources.iris

import org.apache.hadoop.conf.{Configuration => HadoopConfiguration}
import org.apache.hadoop.fs.Path
import org.apache.hadoop.mapreduce._
import org.apache.hadoop.mapreduce.lib.input.{CombineFileInputFormat, CombineFileRecordReader, CombineFileSplit}
import org.apache.spark.Logging
import org.apache.spark.ml.attribute.{NominalAttribute, Attribute}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SQLContext}
import org.deeplearning4j.spark.sql.sources.canova.{CanovaRecordReaderAdapter, CanovaImageVectorizer}
import org.deeplearning4j.spark.sql.sources.lfw.LfwRelation
import org.deeplearning4j.spark.sql.sources.mapreduce.{PrunedReader, ColumnarRecordReader, LabelRecordReader, CachedStatus}
import org.deeplearning4j.spark.sql.types.VectorUDT

/**
 * Iris dataset as a Spark SQL relation.
 *
 * @author Eron Wright
 */
case class IrisRelation(location: String)(@transient val sqlContext: SQLContext)
  extends BaseRelation
  with PrunedScan with Logging {

  private val labelMetadata = new MetadataBuilder().putMetadata("ml_attr",
    new MetadataBuilder().putLong("num_vals", 3).build()).build()

  override def schema: StructType = StructType(
    StructField("label", DoubleType, nullable = false, metadata = labelMetadata) ::
      StructField("features", VectorUDT(), nullable = false) :: Nil)

  override def buildScan(requiredColumns: Array[String]): RDD[Row] = {

    val sc = sqlContext.sparkContext
    val baseRdd = MLUtils.loadLibSVMFile(sc, location)

    val rowBuilders = requiredColumns.map {
      case "label" => (pt: LabeledPoint) => Seq(pt.label)
      case "features" => (pt: LabeledPoint) => Seq(pt.features.toDense)
    }

    baseRdd.map(pt => {
      Row.fromSeq(rowBuilders.map(_(pt)).reduceOption(_ ++ _).getOrElse(Seq.empty))
    })
  }

  override def hashCode(): Int = 41 * (41 + location.hashCode) + schema.hashCode()

  override def equals(other: Any): Boolean = other match {
    case that: IrisRelation =>
      (this.location == that.location) && this.schema.equals(that.schema)
    case _ => false
  }
}

/**
 * Iris dataset provider.
 */
class DefaultSource extends RelationProvider {
  private def checkPath(parameters: Map[String, String]): String = {
    parameters.getOrElse("path", sys.error("'path' must be specified for Iris data."))
  }

  override def createRelation(sqlContext: SQLContext, parameters: Map[String, String]) = {
    val path = checkPath(parameters)
    new IrisRelation(path)(sqlContext)
  }
}





