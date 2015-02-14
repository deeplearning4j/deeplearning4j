package org.nd4j.api.linalg.complex

import org.nd4j.linalg.api.complex.{IComplexFloat, BaseComplexDouble}

/**
 * Created by agibsonccc on 2/13/15.
 */
class SComplexDouble extends BaseComplexDouble {

  def this(real : Double,imag : Double) {
    this()
    this.real = real
    this.imag = imag
  }

  override def asFloat(): IComplexFloat = {
    return new SComplexFloat(real.asInstanceOf,imag.asInstanceOf)
  }
}
