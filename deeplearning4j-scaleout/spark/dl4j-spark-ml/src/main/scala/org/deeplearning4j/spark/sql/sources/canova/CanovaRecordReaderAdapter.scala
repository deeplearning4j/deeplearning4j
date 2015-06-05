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

package org.deeplearning4j.spark.sql.sources.canova

import org.apache.hadoop.fs.FSDataInputStream
import org.apache.hadoop.mapreduce.lib.input.CombineFileSplit
import org.apache.hadoop.mapreduce.{InputSplit => HadoopInputSplit, RecordReader => HadoopRecordReader, TaskAttemptContext}
import org.apache.spark.Logging
import org.apache.spark.sql.Row
import org.canova.api.conf.{Configuration => CanovaConfiguration}
import org.canova.api.records.reader.{RecordReader => CanovaRecordReader}
import org.canova.api.split.InputStreamInputSplit
import org.canova.api.writable.{Writable => CanovaWritable}

/**
 * Adapts a Canova record reader to Hadoop.
 *
 * The subclass provides a concrete Canova reader.
 * Each split is fed to the reader as an input stream.   For CombineFileSplit, the stream
 * corresponds to a single file.
 *
 * @author Eron Wright
 */
abstract class CanovaRecordReaderAdapter(val split: CombineFileSplit, val context: TaskAttemptContext, val index: Integer)
  extends HadoopRecordReader[String,Row]
  with Logging {

  val canovaConf: CanovaConfiguration = new CanovaConfiguration()

  var stream: FSDataInputStream = null

  var value: Row = null.asInstanceOf[Row]

  val reader: CanovaRecordReader

  override def initialize(_split: HadoopInputSplit, context: TaskAttemptContext): Unit = {

    // todo propagate the config
    //context.getConfiguration.forEach{case(e:java.util.Map.Entry[String,String])=>canovaConf.set(e.getKey, e.getValue)}

    val path = split.getPath(index)
    val fs = path.getFileSystem(context.getConfiguration)
    stream = fs.open(path)

    val csplit = new InputStreamInputSplit(stream, path.toUri)
    reader.initialize(canovaConf, csplit)
  }

  override def close(): Unit = {
    if(stream != null) {
      stream.close()
    }
  }

  override def getProgress: Float = if (!reader.hasNext()) 1.0f else 0.0f

  override def getCurrentKey: String = ""

  override def getCurrentValue: Row = value

  override def nextKeyValue: Boolean
}
