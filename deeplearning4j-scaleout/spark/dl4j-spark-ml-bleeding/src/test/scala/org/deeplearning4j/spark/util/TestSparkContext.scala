package org.deeplearning4j.spark.util

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{Suite, BeforeAndAfterAll}

/**
 * A SparkContext for test classes.
 */
trait TestSparkContext extends BeforeAndAfterAll { self: Suite =>

  @transient var sc: SparkContext = _
  @transient var sqlContext: SQLContext = _

  override def beforeAll() {
    val sparkConf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("sparktest")
    sc = new SparkContext(sparkConf)
    sqlContext = new SQLContext(sc)
    super.beforeAll()
  }

  override def afterAll() {
    sqlContext = null
    if( sc != null) {
      sc.stop
    }
    sc = null
    super.afterAll()
  }
}
