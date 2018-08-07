/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4s

import org.nd4j.linalg.api.complex.IComplexNumber

/**
 * Created by taisukeoe on 16/02/12.
 */
trait Equality[A]{
   def equal(left:A,right:A):Boolean
}
object Equality{
  implicit lazy val doubleEquality = new Equality[Double] {
    lazy val tolerance = 0.01D
    override def equal(left: Double, right: Double): Boolean = math.abs(left - right) < tolerance
  }
  implicit lazy val floatEquality = new Equality[Float] {
    lazy val tolerance = 0.01F
    override def equal(left: Float, right: Float): Boolean = math.abs(left - right) < tolerance
  }
  implicit lazy val complexEquality = new Equality[IComplexNumber] {
    lazy val tolerance = 0.01D
    override def equal(left: IComplexNumber, right: IComplexNumber): Boolean = math.abs(left.absoluteValue().doubleValue() - right.absoluteValue().doubleValue()) < tolerance
  }
}
