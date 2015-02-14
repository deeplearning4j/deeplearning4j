package org.nd4j.api.linalg


import org.junit.{Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
 * Created by agibsonccc on 2/13/15.
 */
class TestNDArray extends AssertionsForJUnit {
  @Before
  def before(): Unit = {
    SNd4j.initContext()
  }

  @Test
  def testCreate() {
    var arr = SNd4j.create(5)
    var arr2 = SNd4j.create(5)
    arr = arr + arr2
    arr += 1
    val arrT = arr.T

  }
}
