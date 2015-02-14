package org.nd4j.api.linalg.complex

import org.nd4j.linalg.api.complex.{IComplexNumber, BaseComplexFloat}

/**
 * Created by agibsonccc on 2/13/15.
 */
class SComplexFloat extends BaseComplexFloat {

  def this(real: Float,imag : Float) {
    this()
    this.real = real
    this.imag = imag
  }


  override def dup(): IComplexNumber = {
    return new SComplexFloat(realComponent(),imaginaryComponent())
  }
}
