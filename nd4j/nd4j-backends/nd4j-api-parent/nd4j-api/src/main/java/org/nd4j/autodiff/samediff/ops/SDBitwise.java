/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff.ops;

import static org.nd4j.autodiff.samediff.ops.SDValidation.isSameType;

import java.lang.String;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;

public class SDBitwise extends SDOps {
  public SDBitwise(SameDiff sameDiff) {
    super(sameDiff);
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
  public SDVariable and(SDVariable x, SDVariable y) {
    SDValidation.validateInteger("and", "x", x);
    SDValidation.validateInteger("and", "y", y);
    Preconditions.checkArgument(isSameType(x, y), "Must be same types");
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.BitwiseAnd(sd,x, y).outputVariable();
  }

  /**
   * Bitwise AND operation. Supports broadcasting.<br>
   *
   * Inputs must satisfy the following constraints: <br>
   * Must be same types: isSameType(x, y)<br>
   * Must have broadcastable shapes: isBroadcastableShapes(x, y)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x First input array (INT type)
   * @param y Second input array (INT type)
   * @return output Bitwise AND array (INT type)
   */
  public SDVariable and(String name, SDVariable x, SDVariable y) {
    SDValidation.validateInteger("and", "x", x);
    SDValidation.validateInteger("and", "y", y);
    Preconditions.checkArgument(isSameType(x, y), "Must be same types");
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.BitwiseAnd(sd,x, y).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Roll integer bits to the left, i.e. var << 4 | var >> (32 - 4)<br>
   *
   * @param x Input 1 (INT type)
   * @param shift Number of bits to shift. (INT type)
   * @return output SDVariable with shifted bits (INT type)
   */
  public SDVariable bitRotl(SDVariable x, SDVariable shift) {
    SDValidation.validateInteger("bitRotl", "x", x);
    SDValidation.validateInteger("bitRotl", "shift", shift);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicShiftBits(sd,x, shift).outputVariable();
  }

  /**
   * Roll integer bits to the left, i.e. var << 4 | var >> (32 - 4)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input 1 (INT type)
   * @param shift Number of bits to shift. (INT type)
   * @return output SDVariable with shifted bits (INT type)
   */
  public SDVariable bitRotl(String name, SDVariable x, SDVariable shift) {
    SDValidation.validateInteger("bitRotl", "x", x);
    SDValidation.validateInteger("bitRotl", "shift", shift);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicShiftBits(sd,x, shift).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Roll integer bits to the right, i.e. var >> 4 | var << (32 - 4)<br>
   *
   * @param x Input 1 (INT type)
   * @param shift Number of bits to shift. (INT type)
   * @return output SDVariable with shifted bits (INT type)
   */
  public SDVariable bitRotr(SDVariable x, SDVariable shift) {
    SDValidation.validateInteger("bitRotr", "x", x);
    SDValidation.validateInteger("bitRotr", "shift", shift);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicRShiftBits(sd,x, shift).outputVariable();
  }

  /**
   * Roll integer bits to the right, i.e. var >> 4 | var << (32 - 4)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input 1 (INT type)
   * @param shift Number of bits to shift. (INT type)
   * @return output SDVariable with shifted bits (INT type)
   */
  public SDVariable bitRotr(String name, SDVariable x, SDVariable shift) {
    SDValidation.validateInteger("bitRotr", "x", x);
    SDValidation.validateInteger("bitRotr", "shift", shift);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicRShiftBits(sd,x, shift).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Shift integer bits to the left, i.e. var << 4<br>
   *
   * @param x Input 1 (INT type)
   * @param shift Number of bits to shift. (INT type)
   * @return output SDVariable with shifted bits (INT type)
   */
  public SDVariable bitShift(SDVariable x, SDVariable shift) {
    SDValidation.validateInteger("bitShift", "x", x);
    SDValidation.validateInteger("bitShift", "shift", shift);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.ShiftBits(sd,x, shift).outputVariable();
  }

  /**
   * Shift integer bits to the left, i.e. var << 4<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input 1 (INT type)
   * @param shift Number of bits to shift. (INT type)
   * @return output SDVariable with shifted bits (INT type)
   */
  public SDVariable bitShift(String name, SDVariable x, SDVariable shift) {
    SDValidation.validateInteger("bitShift", "x", x);
    SDValidation.validateInteger("bitShift", "shift", shift);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.ShiftBits(sd,x, shift).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Shift integer bits to the right, i.e. var >> 4<br>
   *
   * @param x Input 1 (INT type)
   * @param shift Number of bits to shift. (INT type)
   * @return output SDVariable with shifted bits (INT type)
   */
  public SDVariable bitShiftRight(SDVariable x, SDVariable shift) {
    SDValidation.validateInteger("bitShiftRight", "x", x);
    SDValidation.validateInteger("bitShiftRight", "shift", shift);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.RShiftBits(sd,x, shift).outputVariable();
  }

  /**
   * Shift integer bits to the right, i.e. var >> 4<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input 1 (INT type)
   * @param shift Number of bits to shift. (INT type)
   * @return output SDVariable with shifted bits (INT type)
   */
  public SDVariable bitShiftRight(String name, SDVariable x, SDVariable shift) {
    SDValidation.validateInteger("bitShiftRight", "x", x);
    SDValidation.validateInteger("bitShiftRight", "shift", shift);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.RShiftBits(sd,x, shift).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable bitsHammingDistance(SDVariable x, SDVariable y) {
    SDValidation.validateInteger("bitsHammingDistance", "x", x);
    SDValidation.validateInteger("bitsHammingDistance", "y", y);
    Preconditions.checkArgument(isSameType(x, y), "Must be same types");
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.BitsHammingDistance(sd,x, y).outputVariable();
  }

  /**
   * Bitwise Hamming distance reduction over all elements of both input arrays.<br>
   * For example, if x=01100000 and y=1010000 then the bitwise Hamming distance is 2 (due to differences at positions 0 and 1)<br>
   *
   * Inputs must satisfy the following constraints: <br>
   * Must be same types: isSameType(x, y)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x First input array. (INT type)
   * @param y Second input array. (INT type)
   * @return output bitwise Hamming distance (INT type)
   */
  public SDVariable bitsHammingDistance(String name, SDVariable x, SDVariable y) {
    SDValidation.validateInteger("bitsHammingDistance", "x", x);
    SDValidation.validateInteger("bitsHammingDistance", "y", y);
    Preconditions.checkArgument(isSameType(x, y), "Must be same types");
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.BitsHammingDistance(sd,x, y).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Bitwise left shift operation. Supports broadcasting.<br>
   *
   * @param x Input to be bit shifted (INT type)
   * @param y Amount to shift elements of x array (INT type)
   * @return output Bitwise shifted input x (INT type)
   */
  public SDVariable leftShift(SDVariable x, SDVariable y) {
    SDValidation.validateInteger("leftShift", "x", x);
    SDValidation.validateInteger("leftShift", "y", y);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.ShiftBits(sd,x, y).outputVariable();
  }

  /**
   * Bitwise left shift operation. Supports broadcasting.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input to be bit shifted (INT type)
   * @param y Amount to shift elements of x array (INT type)
   * @return output Bitwise shifted input x (INT type)
   */
  public SDVariable leftShift(String name, SDVariable x, SDVariable y) {
    SDValidation.validateInteger("leftShift", "x", x);
    SDValidation.validateInteger("leftShift", "y", y);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.ShiftBits(sd,x, y).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable leftShiftCyclic(SDVariable x, SDVariable y) {
    SDValidation.validateInteger("leftShiftCyclic", "x", x);
    SDValidation.validateInteger("leftShiftCyclic", "y", y);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicShiftBits(sd,x, y).outputVariable();
  }

  /**
   * Bitwise left cyclical shift operation. Supports broadcasting.<br>
   * Unlike {@link #leftShift(INDArray, INDArray)} the bits will "wrap around":<br>
   * {@code leftShiftCyclic(01110000, 2) -> 11000001}<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input to be bit shifted (INT type)
   * @param y Amount to shift elements of x array (INT type)
   * @return output Bitwise cyclic shifted input x (INT type)
   */
  public SDVariable leftShiftCyclic(String name, SDVariable x, SDVariable y) {
    SDValidation.validateInteger("leftShiftCyclic", "x", x);
    SDValidation.validateInteger("leftShiftCyclic", "y", y);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicShiftBits(sd,x, y).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable or(SDVariable x, SDVariable y) {
    SDValidation.validateInteger("or", "x", x);
    SDValidation.validateInteger("or", "y", y);
    Preconditions.checkArgument(isSameType(x, y), "Must be same types");
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.BitwiseOr(sd,x, y).outputVariable();
  }

  /**
   * Bitwise OR operation. Supports broadcasting.<br>
   *
   * Inputs must satisfy the following constraints: <br>
   * Must be same types: isSameType(x, y)<br>
   * Must have broadcastable shapes: isBroadcastableShapes(x, y)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x First input array (INT type)
   * @param y First input array (INT type)
   * @return output Bitwise OR array (INT type)
   */
  public SDVariable or(String name, SDVariable x, SDVariable y) {
    SDValidation.validateInteger("or", "x", x);
    SDValidation.validateInteger("or", "y", y);
    Preconditions.checkArgument(isSameType(x, y), "Must be same types");
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.BitwiseOr(sd,x, y).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Bitwise right shift operation. Supports broadcasting. <br>
   *
   * @param x Input to be bit shifted (INT type)
   * @param y Amount to shift elements of x array (INT type)
   * @return output Bitwise shifted input x (INT type)
   */
  public SDVariable rightShift(SDVariable x, SDVariable y) {
    SDValidation.validateInteger("rightShift", "x", x);
    SDValidation.validateInteger("rightShift", "y", y);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.RShiftBits(sd,x, y).outputVariable();
  }

  /**
   * Bitwise right shift operation. Supports broadcasting. <br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input to be bit shifted (INT type)
   * @param y Amount to shift elements of x array (INT type)
   * @return output Bitwise shifted input x (INT type)
   */
  public SDVariable rightShift(String name, SDVariable x, SDVariable y) {
    SDValidation.validateInteger("rightShift", "x", x);
    SDValidation.validateInteger("rightShift", "y", y);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.RShiftBits(sd,x, y).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable rightShiftCyclic(SDVariable x, SDVariable y) {
    SDValidation.validateInteger("rightShiftCyclic", "x", x);
    SDValidation.validateInteger("rightShiftCyclic", "y", y);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicRShiftBits(sd,x, y).outputVariable();
  }

  /**
   * Bitwise right cyclical shift operation. Supports broadcasting.<br>
   * Unlike {@link #rightShift(INDArray, INDArray)} the bits will "wrap around":<br>
   * {@code rightShiftCyclic(00001110, 2) -> 10000011}<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input to be bit shifted (INT type)
   * @param y Amount to shift elements of x array (INT type)
   * @return output Bitwise cyclic shifted input x (INT type)
   */
  public SDVariable rightShiftCyclic(String name, SDVariable x, SDVariable y) {
    SDValidation.validateInteger("rightShiftCyclic", "x", x);
    SDValidation.validateInteger("rightShiftCyclic", "y", y);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicRShiftBits(sd,x, y).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable xor(SDVariable x, SDVariable y) {
    SDValidation.validateInteger("xor", "x", x);
    SDValidation.validateInteger("xor", "y", y);
    Preconditions.checkArgument(isSameType(x, y), "Must be same types");
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.BitwiseXor(sd,x, y).outputVariable();
  }

  /**
   * Bitwise XOR operation (exclusive OR). Supports broadcasting.<br>
   *
   * Inputs must satisfy the following constraints: <br>
   * Must be same types: isSameType(x, y)<br>
   * Must have broadcastable shapes: isBroadcastableShapes(x, y)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x First input array (INT type)
   * @param y First input array (INT type)
   * @return output Bitwise XOR array (INT type)
   */
  public SDVariable xor(String name, SDVariable x, SDVariable y) {
    SDValidation.validateInteger("xor", "x", x);
    SDValidation.validateInteger("xor", "y", y);
    Preconditions.checkArgument(isSameType(x, y), "Must be same types");
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.BitwiseXor(sd,x, y).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }
}
