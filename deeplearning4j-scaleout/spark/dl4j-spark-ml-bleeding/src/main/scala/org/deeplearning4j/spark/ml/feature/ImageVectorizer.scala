package org.deeplearning4j.spark.ml.feature

import java.io.ByteArrayInputStream

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.{Param, IntParam, ParamValidators}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.types.{BinaryType, DataType}
import org.deeplearning4j.spark.ml.util.Identifiable
import org.canova.image.loader.ImageLoader
import org.deeplearning4j.spark.sql.types.VectorUDT
import org.deeplearning4j.spark.util.conversions.toVector

/**
 * Vectorize images stored in a binary DataFrame column
 */
class ImageVectorizer(override val uid: String) extends UnaryTransformer[Array[Byte], Vector, ImageVectorizer] {
  def this() = this(Identifiable.randomUID("imageVectorizer"))

  val height: Param[Int] = new IntParam(this, "height", "image height", ParamValidators.gtEq(0))
  val width: Param[Int] = new IntParam(this, "width", "image width", ParamValidators.gtEq(0))
  val channels: Param[Int] = new IntParam(this, "channels", "number of channels", ParamValidators.gtEq(0))

  setDefault(height -> 28)
  setDefault(width -> 28)
  setDefault(channels -> 3)

  def setHeight(value: Int): this.type = set(height, value)
  def setWidth(value: Int): this.type = set(width, value)
  def setChannels(value: Int): this.type = set(channels, value)

  def getHeight: Int = $(height)
  def getWidth: Int = $(width)
  def getChannels: Int = $(channels)

  override protected def createTransformFunc: Array[Byte] => Vector = {
    val imageLoader = new ImageLoader($(height), $(width), $(channels))
    content: Array[Byte] =>
      val imgStream = new ByteArrayInputStream(content)
      val vector = imageLoader.asRowVector(imgStream)
      println(vector.length())
      toVector(vector)
  }

  override protected def validateInputType(inputType: DataType): Unit = {
    require(inputType == BinaryType,
      s"Input type must be BinaryType but got $inputType.")
  }

  override protected def outputDataType: DataType = VectorUDT()
}