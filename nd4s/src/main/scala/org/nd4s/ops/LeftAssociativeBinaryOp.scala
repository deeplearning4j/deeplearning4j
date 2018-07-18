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

import org.nd4j.linalg.api.complex.IComplexNumber
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.{BaseScalarOp, Op}
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

trait LeftAssociativeBinaryOp {

  def op(origin: IComplexNumber, other: Double): IComplexNumber = op(origin)

  def op(origin: IComplexNumber, other: Float): IComplexNumber = op(origin)

  def op(origin: IComplexNumber, other: IComplexNumber): IComplexNumber = op(origin)

  def op(origin: Float, other: Float): Float = op(origin)

  def op(origin: Double, other: Double): Double = op(origin)

  def op(origin: Double): Double

  def op(origin: Float): Float

  def op(origin: IComplexNumber): IComplexNumber
}
