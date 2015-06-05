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

package org.deeplearning4j.spark.sql.sources.mapreduce

import java.util.{List => JavaList}

import org.apache.hadoop.conf.{Configuration => HadoopConfiguration}
import org.apache.hadoop.fs.FileStatus
import org.apache.hadoop.mapreduce._
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.spark.sql.Row

import scala.collection.mutable

/**
 * An input format with cached file status.
 */
trait CachedStatus extends FileInputFormat[String, Row] {
  private val statusCache = new mutable.WeakHashMap[JobID,JavaList[FileStatus]]()

  override def listStatus(job: JobContext): JavaList[FileStatus] = {
    statusCache.getOrElseUpdate(job.getJobID, {
      super.listStatus(job)
    })
  }
}

/**
 * A record reader with pruned columns.
 */
trait PrunedReader extends RecordReader[String,Row] {

  protected var requiredColumns: Array[String] = Array()

  abstract override def initialize(split: InputSplit, context: TaskAttemptContext): Unit = {
    requiredColumns = PrunedReader.getRequiredColumns(context.getConfiguration)
    super.initialize(split, context)
  }
}

object PrunedReader {
  val REQUIRED_COLUMNS = "org.deeplearning4j.spark.sql.sources.mapreduce.PrunedReader.REQUIRED_COLUMNS"

  def setRequiredColumns(conf: HadoopConfiguration, requiredColumns: Array[String]): Unit = {
    conf.setStrings(REQUIRED_COLUMNS, requiredColumns: _*)
  }

  def getRequiredColumns(conf: HadoopConfiguration): Array[String] = {
    conf.getStrings(REQUIRED_COLUMNS)
  }
}
