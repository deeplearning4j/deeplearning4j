package org.nd4j.api.linalg

import org.junit.Before
import org.nd4j.linalg.api.test.ComplexNDArrayTests

/**
 * Created by agibsonccc on 2/14/15.
 */
class SComplexNDArrayTests extends ComplexNDArrayTests {
  @Before
  override def before(): Unit = {
    SNd4j.initContext()
    super.before()
  }
}
