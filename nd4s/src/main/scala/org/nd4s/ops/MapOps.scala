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

package org.nd4s.ops

import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.linalg.api.complex.IComplexNumber
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.{BaseScalarOp, BaseOp, Op}
import org.nd4s.Implicits._

object MapOps{
  def apply(x:INDArray,f:Double=>Double,g:IComplexNumber =>IComplexNumber):MapOps = new MapOps(x,f,g)
}
class MapOps(_x:INDArray,f:Double => Double, g:IComplexNumber => IComplexNumber) extends BaseScalarOp(_x,null,_x,_x.length(),0) with LeftAssociativeBinaryOp {
  x = _x
  def this(){
    this(0.toScalar,null,null)
  }

  override def opNum(): Int = -1

  override def opName(): String = "map_scalar"

  override def onnxName(): String = throw new UnsupportedOperationException

  override def tensorflowName(): String = throw new UnsupportedOperationException

  override def doDiff(f1: java.util.List[SDVariable]): java.util.List[SDVariable] = throw new UnsupportedOperationException

//  override def opForDimension(index: Int, dimension: Int): Op = MapOps(x.tensorAlongDimension(index,dimension),f,g)
//
//  override def opForDimension(index: Int, dimension: Int*): Op = MapOps(x.tensorAlongDimension(index,dimension:_*),f,g)

  override def op(origin: Double): Double = f(origin)

  override def op(origin: Float): Float = f(origin).toFloat

  override def op(origin: IComplexNumber): IComplexNumber = g(origin)
}
