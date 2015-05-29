package org.apache.spark.sql.sources.mapreduce

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
  val REQUIRED_COLUMNS = "org.apache.spark.sql.sources.mapreduce.PrunedReader.REQUIRED_COLUMNS"

  def setRequiredColumns(conf: HadoopConfiguration, requiredColumns: Array[String]): Unit = {
    conf.setStrings(REQUIRED_COLUMNS, requiredColumns: _*)
  }

  def getRequiredColumns(conf: HadoopConfiguration): Array[String] = {
    conf.getStrings(REQUIRED_COLUMNS)
  }
}
