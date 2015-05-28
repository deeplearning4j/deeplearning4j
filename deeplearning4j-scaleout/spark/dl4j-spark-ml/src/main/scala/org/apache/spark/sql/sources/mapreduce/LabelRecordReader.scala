package org.apache.spark.sql.sources.mapreduce

import org.apache.hadoop.mapreduce.lib.input.CombineFileSplit
import org.apache.hadoop.mapreduce.{InputSplit, RecordReader, TaskAttemptContext}
import org.apache.spark.sql.Row

/**
 * A record reader that produces labels based on the filesystem path.
 *
 * Produces a single row for each path in the split.
 */
class LabelRecordReader(val split: CombineFileSplit, val context: TaskAttemptContext, val index: Integer)
  extends RecordReader[String,Row] {

  var value: Row = null

  var processed = false

  override def initialize(split: InputSplit, context: TaskAttemptContext): Unit = {}

  override def getProgress: Float = if (processed) 1.0f else 0.0f

  override def nextKeyValue(): Boolean = {
    if(processed) return false
    processed = true
    value = Row(split.getPath(index).getParent.getName)
    true
  }

  override def getCurrentKey: String = ""

  override def getCurrentValue: Row = value

  override def close(): Unit = {}
}
