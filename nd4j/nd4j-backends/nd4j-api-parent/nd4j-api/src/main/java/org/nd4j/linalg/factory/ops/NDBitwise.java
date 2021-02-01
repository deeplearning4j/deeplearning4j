/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.nd4j.linalg.factory.ops;

import static org.nd4j.linalg.factory.NDValidation.isSameType;

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.factory.Nd4j;

public class NDBitwise {
  public NDBitwise() {
  }

  /**
   * Bitwise AND operation. Supports broadcasting.<br>
   *
   * Inputs must satisfy the following constraints: <br>
   * Must be same types: isSameType(x, y)<br>
   * Must have broadcastable shapes: isBroadcastableShapes(x, y)<br>
   *
   * @param x First input array (INT type)
   * @param y Second input array (INT type)
   * @return output Bitwise AND array (INT type)
   */
  public INDArray and(INDArray x, INDArray y) {
    NDValidation.validateInteger("and", "x", x);
    NDValidation.validateInteger("and", "y", y);
    Preconditions.checkArgument(isSameType(x, y), "Must be same types");
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.BitwiseAnd(x, y))[0];
  }

  /**
   * Roll integer bits to the left, i.e. var << 4 | var >> (32 - 4)<br>
   *
   * @param x Input 1 (INT type)
   * @param shift Number of bits to shift. (INT type)
   * @return output SDVariable with shifted bits (INT type)
   */
  public INDArray bitRotl(INDArray x, INDArray shift) {
    NDValidation.validateInteger("bitRotl", "x", x);
    NDValidation.validateInteger("bitRotl", "shift", shift);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicShiftBits(x, shift))[0];
  }

  /**
   * Roll integer bits to the right, i.e. var >> 4 | var << (32 - 4)<br>
   *
   * @param x Input 1 (INT type)
   * @param shift Number of bits to shift. (INT type)
   * @return output SDVariable with shifted bits (INT type)
   */
  public INDArray bitRotr(INDArray x, INDArray shift) {
    NDValidation.validateInteger("bitRotr", "x", x);
    NDValidation.validateInteger("bitRotr", "shift", shift);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicRShiftBits(x, shift))[0];
  }

  /**
   * Shift integer bits to the left, i.e. var << 4<br>
   *
   * @param x Input 1 (INT type)
   * @param shift Number of bits to shift. (INT type)
   * @return output SDVariable with shifted bits (INT type)
   */
  public INDArray bitShift(INDArray x, INDArray shift) {
    NDValidation.validateInteger("bitShift", "x", x);
    NDValidation.validateInteger("bitShift", "shift", shift);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.ShiftBits(x, shift))[0];
  }

  /**
   * Shift integer bits to the right, i.e. var >> 4<br>
   *
   * @param x Input 1 (INT type)
   * @param shift Number of bits to shift. (INT type)
   * @return output SDVariable with shifted bits (INT type)
   */
  public INDArray bitShiftRight(INDArray x, INDArray shift) {
    NDValidation.validateInteger("bitShiftRight", "x", x);
    NDValidation.validateInteger("bitShiftRight", "shift", shift);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.RShiftBits(x, shift))[0];
  }

  /**
   * Bitwise Hamming distance reduction over all elements of both input arrays.<br>
   * For example, if x=01100000 and y=1010000 then the bitwise Hamming distance is 2 (due to differences at positions 0 and 1)<br>
   *
   * Inputs must satisfy the following constraints: <br>
   * Must be same types: isSameType(x, y)<br>
   *
   * @param x First input array. (INT type)
   * @param y Second input array. (INT type)
   * @return output bitwise Hamming distance (INT type)
   */
  public INDArray bitsHammingDistance(INDArray x, INDArray y) {
    NDValidation.validateInteger("bitsHammingDistance", "x", x);
    NDValidation.validateInteger("bitsHammingDistance", "y", y);
    Preconditions.checkArgument(isSameType(x, y), "Must be same types");
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.BitsHammingDistance(x, y))[0];
  }

  /**
   * Bitwise left shift operation. Supports broadcasting.<br>
   *
   * @param x Input to be bit shifted (INT type)
   * @param y Amount to shift elements of x array (INT type)
   * @return output Bitwise shifted input x (INT type)
   */
  public INDArray leftShift(INDArray x, INDArray y) {
    NDValidation.validateInteger("leftShift", "x", x);
    NDValidation.validateInteger("leftShift", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.ShiftBits(x, y))[0];
  }

  /**
   * Bitwise left cyclical shift operation. Supports broadcasting.<br>
   * Unlike {@link #leftShift(INDArray, INDArray)} the bits will "wrap around":<br>
   * {@code leftShiftCyclic(01110000, 2) -> 11000001}<br>
   *
   * @param x Input to be bit shifted (INT type)
   * @param y Amount to shift elements of x array (INT type)
   * @return output Bitwise cyclic shifted input x (INT type)
   */
  public INDArray leftShiftCyclic(INDArray x, INDArray y) {
    NDValidation.validateInteger("leftShiftCyclic", "x", x);
    NDValidation.validateInteger("leftShiftCyclic", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicShiftBits(x, y))[0];
  }

  /**
   * Bitwise OR operation. Supports broadcasting.<br>
   *
   * Inputs must satisfy the following constraints: <br>
   * Must be same types: isSameType(x, y)<br>
   * Must have broadcastable shapes: isBroadcastableShapes(x, y)<br>
   *
   * @param x First input array (INT type)
   * @param y First input array (INT type)
   * @return output Bitwise OR array (INT type)
   */
  public INDArray or(INDArray x, INDArray y) {
    NDValidation.validateInteger("or", "x", x);
    NDValidation.validateInteger("or", "y", y);
    Preconditions.checkArgument(isSameType(x, y), "Must be same types");
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.BitwiseOr(x, y))[0];
  }

  /**
   * Bitwise right shift operation. Supports broadcasting. <br>
   *
   * @param x Input to be bit shifted (INT type)
   * @param y Amount to shift elements of x array (INT type)
   * @return output Bitwise shifted input x (INT type)
   */
  public INDArray rightShift(INDArray x, INDArray y) {
    NDValidation.validateInteger("rightShift", "x", x);
    NDValidation.validateInteger("rightShift", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.RShiftBits(x, y))[0];
  }

  /**
   * Bitwise right cyclical shift operation. Supports broadcasting.<br>
   * Unlike {@link #rightShift(INDArray, INDArray)} the bits will "wrap around":<br>
   * {@code rightShiftCyclic(00001110, 2) -> 10000011}<br>
   *
   * @param x Input to be bit shifted (INT type)
   * @param y Amount to shift elements of x array (INT type)
   * @return output Bitwise cyclic shifted input x (INT type)
   */
  public INDArray rightShiftCyclic(INDArray x, INDArray y) {
    NDValidation.validateInteger("rightShiftCyclic", "x", x);
    NDValidation.validateInteger("rightShiftCyclic", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicRShiftBits(x, y))[0];
  }

  /**
   * Bitwise XOR operation (exclusive OR). Supports broadcasting.<br>
   *
   * Inputs must satisfy the following constraints: <br>
   * Must be same types: isSameType(x, y)<br>
   * Must have broadcastable shapes: isBroadcastableShapes(x, y)<br>
   *
   * @param x First input array (INT type)
   * @param y First input array (INT type)
   * @return output Bitwise XOR array (INT type)
   */
  public INDArray xor(INDArray x, INDArray y) {
    NDValidation.validateInteger("xor", "x", x);
    NDValidation.validateInteger("xor", "y", y);
    Preconditions.checkArgument(isSameType(x, y), "Must be same types");
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.BitwiseXor(x, y))[0];
  }
}
