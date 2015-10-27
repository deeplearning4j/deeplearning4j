package org.deeplearning4j.spark.sql.sources.iris

import org.apache.spark.Logging
import org.deeplearning4j.spark.util.TestSparkContext
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, FunSuite}
import org.springframework.core.io.ClassPathResource

/**
 * Test IrisRelation
 */
@RunWith(classOf[JUnitRunner])
class IrisRelationTest
  extends FunSuite with TestSparkContext with Logging with Matchers {

  val path = new ClassPathResource("/data/irisSvmLight.txt",
    classOf[IrisRelationTest]).getURI.toString

  test("select") {
    val df = sqlContext.read.iris(path)

    df.count() shouldEqual 12
    df.select("label").count() shouldEqual 12
    df.select("features").count() shouldEqual 12
    df.show(numRows = 3)
  }

  test("sql datasource") {
    sqlContext.sql(
      s"""
         |CREATE TEMPORARY TABLE iris
         |USING org.deeplearning4j.spark.sql.sources.iris
         |OPTIONS (path "$path")
       """.stripMargin)

    val df = sqlContext.sql("SELECT * FROM iris")
    df.count() shouldEqual 12
    df.show(numRows = 3)
  }
}
