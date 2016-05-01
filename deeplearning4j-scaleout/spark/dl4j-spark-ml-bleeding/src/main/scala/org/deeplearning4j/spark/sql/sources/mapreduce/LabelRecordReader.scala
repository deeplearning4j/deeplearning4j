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

import org.apache.hadoop.mapreduce.lib.input.CombineFileSplit
import org.apache.hadoop.mapreduce.{InputSplit, RecordReader, TaskAttemptContext}
import org.apache.spark.sql.Row

/**
 * A record reader that produces labels based on the filesystem path.
 *
 * Produces a single row for each path in the split.
 *
 * @author Eron Wright
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
