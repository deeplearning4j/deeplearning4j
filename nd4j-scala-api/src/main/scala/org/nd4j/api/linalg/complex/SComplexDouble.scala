/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

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
