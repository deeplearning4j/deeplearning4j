package org.deeplearning4j.spark.sql.sources.mnist

import java.io.EOFException
import java.nio.file.Paths
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.{Vector => Vector}
import org.deeplearning4j.datasets.mnist.MnistManager
import org.deeplearning4j.spark.sql.sources.mnist.DefaultSource._
import org.deeplearning4j.spark.util.TestSparkContext
import org.junit.runner.RunWith
import org.nd4j.linalg.util.ArrayUtil
import org.scalatest.{Matchers, FunSuite}
import org.scalatest.junit.JUnitRunner
import org.springframework.core.io.ClassPathResource

/**
 * Test Mnist relation
 */
@RunWith(classOf[JUnitRunner])
class MnistRelationTest
  extends FunSuite with TestSparkContext with Logging with Matchers {

  val labels = new ClassPathResource("/data/t10k-labels-idx1-ubyte", classOf[MnistRelationTest]).getURI
  val images = new ClassPathResource("/data/t10k-images-idx3-ubyte", classOf[MnistRelationTest]).getURI

  def open(): MnistManager = {
    // CAVEAT: MnistManager supports only local files at this time.
    val imagesFile = Paths.get(images).toFile.getAbsolutePath
    val labelsFile = Paths.get(labels).toFile.getAbsolutePath
    new MnistManager(imagesFile, labelsFile)
  }

  def imageStream(implicit manager: MnistManager): Stream[Vector] = {
    def next(): Stream[Vector] = {
      try { Vectors.dense(ArrayUtil.flatten(manager.readImage()).map(_.toDouble)) #:: next() }
      catch { case eof: EOFException => Stream.Empty }
    }
    next()
  }

  def labelStream(implicit manager: MnistManager): Stream[Double] = {
    def next(): Stream[Double] = {
      try { manager.readLabel().toDouble #:: next() }
      catch { case eof: EOFException => Stream.Empty }
    }
    next()
  }

  test("select") {
    val df = sqlContext.read.mnist(images.toString, labels.toString)

    df.rdd.partitions.length shouldEqual 10
    df.count() shouldEqual 10000
    df.select("label").count() shouldEqual 10000
    df.select("features").count() shouldEqual 10000
    df.show(numRows = 3)
  }

  test("sql datasource") {
    sqlContext.sql(
      s"""
        |CREATE TEMPORARY TABLE t10k
        |USING org.deeplearning4j.spark.sql.sources.mnist
        |OPTIONS (imagesPath "$images", labelsPath "$labels")
      """.stripMargin)

    val df = sqlContext.sql("SELECT * FROM t10k")
    df.count() shouldEqual 10000
    df.show(numRows = 3)
  }

  test("content") {
    val df = sqlContext.read.mnist(images.toString, labels.toString)
      .select("label", "features")

    implicit val manager = open()
    try {
      val count = df.count().asInstanceOf[Int]
      count shouldEqual manager.getLabels.getCount
      count shouldEqual manager.getImages.getCount

      (labelStream zip imageStream zip df.rdd.toLocalIterator.toIterable) foreach {
        case ((label, features), row) => {
          //println(s"expected: ($label, ${features.size})")
          row.getDouble(0) shouldEqual label
          row.get(1).asInstanceOf[Vector] shouldEqual features
        }
      }
    }
    finally {
      manager.close()
    }
  }

  test("repeatability") {
    val df = sqlContext.read.mnist(images.toString, labels.toString)

    df.count() shouldEqual 10000
    df.count() shouldEqual 10000
  }

  test("partitioning") {

    val parameters = Map(ImagesPath -> images.toString, LabelsPath -> labels.toString)

    var df = sqlContext.read.format(classOf[DefaultSource].getName)
      .options(parameters.updated(s"${RecordsPerPartition}", "1")).load()

    df.rdd.partitions.length shouldEqual 10000

    df = sqlContext.read.format(classOf[DefaultSource].getName)
      .options(parameters.updated(s"${RecordsPerPartition}", "10000")).load()

    df.rdd.partitions.length shouldEqual 1

    df = sqlContext.read.format(classOf[DefaultSource].getName)
      .options(parameters.updated(s"${RecordsPerPartition}", "99")).load()

    df.rdd.partitions.length shouldEqual 102

    df = sqlContext.read.format(classOf[DefaultSource].getName)
      .options(parameters.updated(s"${RecordsPerPartition}", "10001")).load()

    df.rdd.partitions.length shouldEqual 1
  }
}

