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

import java.util

import org.apache.hadoop.conf.{Configuration => HadoopConfiguration}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.Row
import org.canova.api.conf.{Configuration => CanovaConfiguration}
import org.canova.api.io.data.{DoubleWritable, FloatWritable}
import org.canova.api.records.reader.{RecordReader => CanovaRecordReader}
import org.canova.api.writable.{Writable => CanovaWritable}
import org.canova.image.recordreader.{ImageRecordReader => CanovaImageRecordReader, BaseImageRecordReader}

import scala.collection.JavaConversions._

/**
 * Vectorizes an image file using Canova image reader.
 *
 * @author Eron Wright
 */
trait CanovaImageVectorizer extends CanovaRecordReaderAdapter {

  val reader = new CanovaImageRecordReader

  override def nextKeyValue: Boolean = {
    if(!reader.hasNext) return false
    value = Row(convertToVector(reader.next()))
    true
  }

  def convertToVector(record: util.Collection[CanovaWritable]): Vector = {
    val a = new Array[Double](record.size())
    record.toIndexedSeq.map {_ match {
      case d: DoubleWritable => d.get()
      case f: FloatWritable => f.get().asInstanceOf[Double]
      case _: Any => throw new NotImplementedError("unsupported writable")
    }
    }.copyToArray(a)
    Vectors.dense(a)
  }
}

object CanovaImageVectorizer {

  /**
   * Specify the target width of the image.
   * @param conf
   * @param width
   */
  def setWidth(conf: HadoopConfiguration, width: Int) = {
    conf.setInt(BaseImageRecordReader.WIDTH, width)
  }

  def setChannels(conf: HadoopConfiguration,channels: Int) = {
    conf.setInt(BaseImageRecordReader.CHANNELS,channels)
  }


  /**
   * Specify the target height of the image.
   * @param conf
   * @param height
   */
  def setHeight(conf: HadoopConfiguration, height: Int) = {
    conf.setInt(BaseImageRecordReader.HEIGHT, height)
  }
}


