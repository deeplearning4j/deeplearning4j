/*******************************************************************************
  * Copyright (c) 2015-2019 Skymind, Inc.
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
package org.nd4s.samediff.implicits

import org.nd4j.autodiff.samediff.{ SDIndex, SDVariable, SameDiff }
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.samediff.{ SDIndexWrapper, SDVariableWrapper, SameDiffWrapper }

object Implicits {
  implicit def SameDiffToWrapper(sd: SameDiff): SameDiffWrapper =
    new SameDiffWrapper(sd)

  implicit def SDVariableToWrapper(variable: SDVariable): SDVariableWrapper =
    new SDVariableWrapper(variable)

  implicit def FloatToSDVariable(x: Float)(implicit sd: SameDiff): SDVariableWrapper = {
    val result = new SDVariableWrapper(sd.constant(x))
    result.isScalar = true
    result
  }

  implicit def DoubleToSDVariable(x: Double)(implicit sd: SameDiff): SDVariableWrapper = {
    val result = new SDVariableWrapper(sd.constant(x))
    result.isScalar = true
    result
  }

  implicit def BooleanToSDVariable(x: Boolean)(implicit sd: SameDiff): SDVariableWrapper = {
    val result = new SDVariableWrapper(sd.constant(Nd4j.scalar(x)))
    result.isScalar = true
    result
  }

  implicit def RangeToWrapper(start: Long): SDIndexWrapper = {
    val result = new SDIndexWrapper(start)
    result
  }

  implicit def LongToPoint(x: Long): SDIndex =
    SDIndex.point(x)

  implicit def IntRangeToWrapper(start: Int): SDIndexWrapper = {
    val result = new SDIndexWrapper(start)
    result
  }

  implicit def IntToPoint(x: Int): SDIndex =
    SDIndex.point(x)
}
