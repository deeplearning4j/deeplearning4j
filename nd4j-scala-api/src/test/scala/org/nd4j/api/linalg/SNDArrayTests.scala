package org.nd4j.api.linalg

import org.junit.Before
import org.nd4j.linalg.api.test.NDArrayTests
import org.scalatest.junit.AssertionsForJUnit

/**
 * Created by agibsonccc on 2/14/15.
 */
class SNDArrayTests extends NDArrayTests  {


  @Before
  override def before(): Unit = {
    SNd4j.initContext()
    super.before()
  }

}
