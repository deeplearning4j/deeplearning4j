package org.apache.spark.sql.sources.mapreduce

import org.apache.hadoop.mapreduce.{TaskAttemptContext}
import org.apache.hadoop.mapreduce.{InputSplit => HadoopInputSplit, RecordReader => HadoopRecordReader}
import org.apache.spark.sql.Row

/**
 * Record reader of rows consisting of columns produced by one or more associated readers.
 * @param readers a sequence of readers whose rows are to be zipped (merged).
 */
class ColumnarRecordReader(val readers: (String,HadoopRecordReader[String,Row])*)
  extends HadoopRecordReader[String,Row] {

  private var prunedReaders: Seq[HadoopRecordReader[String,Row]] = Nil

  override def initialize(split: HadoopInputSplit, context: TaskAttemptContext): Unit = {

    val requiredColumns = PrunedReader.getRequiredColumns(context.getConfiguration)
    val readerByColumn = readers.toMap
    prunedReaders = requiredColumns.map{(c) => readerByColumn(c)}

    prunedReaders.foreach{(r) => r.initialize(split, context)}
  }

  override def getProgress: Float = {
    prunedReaders.foldLeft(1f) { case (p,r) => math.min(p, r.getProgress)}
  }

  override def nextKeyValue(): Boolean = {
    // advance all readers, stopping once any reader terminates
    prunedReaders.foldLeft(true){(t,r) => r.nextKeyValue() && t}
  }

  override def getCurrentKey: String = ""

  override def getCurrentValue: Row = {
    // merge (column-wise) the rows produced by each reader
    Row.merge(prunedReaders.map{(r) => r.getCurrentValue()}:_*)
  }

  override def close(): Unit = {
    // note: closes only those readers that were initialized
    prunedReaders.foreach{(r) => r.close()}
  }
}
