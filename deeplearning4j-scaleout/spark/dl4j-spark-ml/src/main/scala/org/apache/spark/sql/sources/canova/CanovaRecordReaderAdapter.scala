package org.apache.spark.sql.sources.canova

import java.util

import org.apache.hadoop.fs.FSDataInputStream
import org.apache.hadoop.mapreduce.TaskAttemptContext
import org.apache.hadoop.mapreduce.lib.input.CombineFileSplit
import org.apache.hadoop.mapreduce.{InputSplit=>HadoopInputSplit,RecordReader=>HadoopRecordReader}
import org.apache.spark.Logging
import org.apache.spark.sql.Row
import org.canova.api.split.InputStreamInputSplit
import org.canova.api.records.reader.{RecordReader=>CanovaRecordReader}
import org.canova.api.writable.{Writable=>CanovaWritable}
import org.canova.api.conf.{Configuration => CanovaConfiguration}
import org.canova.api.io.data.{DoubleWritable, FloatWritable}
import org.canova.api.split.InputStreamInputSplit

import scala.reflect.ClassTag



/**
 * Adapts a Canova reader to Hadoop.
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
