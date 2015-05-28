package org.apache.spark.sql.sources.canova

import java.util

import org.apache.hadoop.conf.{Configuration => HadoopConfiguration}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.Row
import org.canova.api.conf.{Configuration => CanovaConfiguration}
import org.canova.api.io.data.{DoubleWritable, FloatWritable}
import org.canova.api.records.reader.{RecordReader => CanovaRecordReader}
import org.canova.api.writable.{Writable => CanovaWritable}
import org.canova.image.recordreader.{ImageRecordReader => CanovaImageRecordReader}

import scala.collection.JavaConversions._


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

  def setWidth(conf: HadoopConfiguration, width: Int) = {
    conf.setInt(CanovaImageRecordReader.WIDTH, width)
  }
  def setHeight(conf: HadoopConfiguration, height: Int) = {
    conf.setInt(CanovaImageRecordReader.HEIGHT, height)
  }
}


/**
 * Default image extractor based on ImageIO.
 * Assumes that all input images are the same dimensions.
 */
//class DefaultImageExtractor
//  extends HadoopFileTransformer[Vector, DefaultImageExtractor] {
//
//  protected def outputDataType = new VectorUDT
//
//  protected def createTransformFunc(dataset: DataFrame, paramMap: ParamMap): HadoopFile => Vector = {
//
//    stream(dataset.sqlContext, (stream: FSDataInputStream) => {
//      val istream = ImageIO.createImageInputStream(stream)
//      try {
//        val ireader = ImageIO.getImageReaders(istream).find(_ => true).get
//
//        val param = ireader.getDefaultReadParam()
//        //        dimension match {
//        //          case Some(d) =>
//        //            if (param.canSetSourceRenderSize) {
//        //              param.setSourceRenderSize(d)
//        //            }
//        //            else throw new NotImplementedError("resize for provided image type")
//        //          case None =>
//        //        }
//
//        ireader.setInput(istream)
//        val image: BufferedImage = ireader.read(0, param)
//
//        val rgbArray = new Array[Int](image.getWidth * image.getHeight)
//        image.getRGB(0, 0, image.getWidth, image.getHeight, rgbArray, 0, image.getWidth)
//        Vectors.dense(rgbArray.map(_.asInstanceOf[Float].asInstanceOf[Double]))
//      }
//      finally {
//        istream.close()
//      }
//    })
//  }
//}
//


