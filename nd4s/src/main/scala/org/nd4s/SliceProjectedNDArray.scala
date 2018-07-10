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

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.{SpecifiedIndex, NDArrayIndex}

class SliceProjectedNDArray(val array:INDArray,filtered:Array[Int]){
  def this(ndarray: INDArray){
    this(ndarray,(0 until ndarray.slices()).toArray)
  }

  def mapi(f:INDArray => INDArray):INDArray = {
    for{
      i <- filtered
    } array.putSlice(i,f(array.slice(i)))
    array.get(new SpecifiedIndex(filtered:_*) +: NDArrayIndex.allFor(array).init:_*)
  }

  def map(f:INDArray => INDArray):INDArray = new SliceProjectedNDArray(array.dup(),filtered).flatMapi(f)

  def flatMap(f:INDArray => INDArray):INDArray = map(f)

  def flatMapi(f:INDArray => INDArray):INDArray = mapi(f)

  def foreach(f:INDArray => Unit):Unit =
    for{
      i <- filtered
    } f(array.slice(i))

  def withFilter(f:INDArray => Boolean):SliceProjectedNDArray = {
    val targets = for{
      i <- filtered
      if f(array.slice(i))
    } yield i
    new SliceProjectedNDArray(array,targets)
  }
}
