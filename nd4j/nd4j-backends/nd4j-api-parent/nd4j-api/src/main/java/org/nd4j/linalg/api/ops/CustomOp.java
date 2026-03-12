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

package org.nd4j.linalg.api.ops;

import lombok.val;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

public interface CustomOp  {

 String getOwnName();

 /**
  * This allows a custom op to configure relevant fields from its arguments.
  * This is needed when ops are created via reflection for things like model import.
  *
  */
 void configureFromArguments();

 /**
  * This method returns op opName as string
  * @return
  */
 String opName();

 /**
  * This method returns LongHash of the opName()
  * @return
  */
 long opHash();

 /**
  * This method returns true if op is supposed to be executed inplace
  * @return
  */
 boolean isInplaceCall();

 List<INDArray> outputArguments();

 List<INDArray> inputArguments();

 long[] iArgs();

 double[] tArgs();

 boolean[] bArgs();

 DataType[] dArgs();

 void addTArgument(double... arg);

 String[] sArgs();

 void addIArgument(int... arg);

 void addIArgument(long... arg);

 void addBArgument(boolean... arg);

 void addDArgument(DataType... arg);

 void removeIArgument(Integer arg);

 void addSArgument(String ... args);

 void removeSArgument(String argument);

 String getSArgument(int index);

 Boolean getBArgument(int index);

 Long getIArgument(int index);

 int numIArguments();

 void removeTArgument(Double arg);

 Double getTArgument(int index);

 int numTArguments();

 int numBArguments();

 int numDArguments();

 int numSArguments();

 void addInputArgument(INDArray... arg);

 void removeInputArgument(INDArray arg);

 INDArray getInputArgument(int index);

 int numInputArguments();


 void addOutputArgument(INDArray... arg);

 void removeOutputArgument(INDArray arg);

 INDArray getOutputArgument(int index);

 int numOutputArguments();


 /**
  * Calculate the output shape for this op
  *
  * @return Output array shapes
  */
 List<DataBuffer> calculateOutputShape();

 /**
  * Calculate the output shape for this op
  *
  * @return Output array shapes
  */
 List<DataBuffer> calculateOutputShape(OpContext opContext);

 /**
  * Get the custom op descriptor if one is available.
  * @return
  */
 CustomOpDescriptor getDescriptor();

 /**
  * Asserts a valid state for execution,
  * otherwise throws an {@link org.nd4j.linalg.exception.ND4JIllegalStateException}
  */
 void assertValidForExecution();

 /**
  * Clear the input and output INDArrays, if any are set
  */
 void clearArrays();

 /**
  * Return true if this op doesn't fully write its output buffers.
  * Such ops (where, scatter_nd, unique, etc.) require pre-zeroed buffers
  * to avoid stale data corruption from cached buffer reuse.
  *
  * Most ops fully write their output and can skip zeroing for performance.
  * Only override this to return true for ops with sparse output patterns.
  */
 default boolean requiresZeroedOutput() {
  return false;
 }


 default void setupOpContextFromCustomOp(OpContext opContext) {
  // Set input arguments
  val inputArgs = inputArguments();
  if (inputArgs != null && !inputArgs.isEmpty()) {
   opContext.setInputArrays(inputArgs.toArray(new INDArray[0]));
  }

  // Set integer arguments
  val iArgs = iArgs();
  if (iArgs != null && iArgs.length > 0) {
   opContext.setIArguments(iArgs);
  }

  // Set floating point arguments
  val tArgs = tArgs();
  if (tArgs != null && tArgs.length > 0) {
   opContext.setTArguments(tArgs);
  }

  // Set boolean arguments
  val bArgs = bArgs();
  if (bArgs != null && bArgs.length > 0) {
   opContext.setBArguments(bArgs);
  }

  // Set data type arguments
  val dArgs = dArgs();
  if (dArgs != null && dArgs.length > 0) {
   opContext.setDArguments(dArgs);
  }

  // Set string arguments
  if (numSArguments() > 0) {
   String[] sArgs = new String[numSArguments()];
   for (int i = 0; i < numSArguments(); i++) {
    sArgs[i] = getSArgument(i);
   }
   opContext.setSArguments(sArgs);
  }

  // Set output arguments
  val outputArgs = outputArguments();
  if (outputArgs != null && !outputArgs.isEmpty()) {
   opContext.setOutputArrays(outputArgs.toArray(new INDArray[0]));
  }
 }

 /**
  * Initialize the output arrays, if required.
  * This method checks for view flags (ARRAY_COPY_OFFSET_INPUT_X) set by the C++ shape function.
  * When a view flag is set, the output is created as a view sharing the input's buffer and offset
  * instead of allocating a new buffer.
  * @return True if the output arrays were initialized (and hence should be calculated), false otherwise
  */
 default boolean initializeOutputs(OpContext ctx) {
  boolean shapeOverride = false;
  if (numOutputArguments() == 0 && !isInplaceCall()) {
   try {
    val list = Nd4j.getExecutioner().calculateOutputShape(this,ctx);
    if (list.isEmpty())
     throw new ND4JIllegalStateException("Op name " + opName() + " failed to calculate output shape and data types.");

    List<INDArray> inputs = inputArguments();
    for (DataBuffer shapeBuffer : list) {
     long[] shapeInfo = shapeBuffer.asLong();

     // Check if a copy offset flag is set (indicates view should be created)
     int copyOffsetInputIndex = ArrayOptionsHelper.getCopyOffsetInputIndex(shapeInfo);

     if (copyOffsetInputIndex >= 0 && inputs != null && copyOffsetInputIndex < inputs.size()) {
      // View case: create output sharing the specified input's buffer and offset
      INDArray input = inputs.get(copyOffsetInputIndex);
      long[] shape = Shape.shape(shapeInfo);
      long[] strides = Shape.stride(shapeInfo);
      char ordering = Shape.order(shapeInfo);

      // Create output as a view sharing the input's buffer at the input's offset
      INDArray newOut = Nd4j.create(input.data(), shape, strides, input.offset(), ordering, true);
      addOutputArgument(newOut);
     } else {
      // Standard case: create new array
      INDArray newOut = Nd4j.createFromDescriptor(shapeBuffer);
      addOutputArgument(newOut);
     }
    }
    shapeOverride = true;
   } catch (ND4JIllegalStateException e) {
    throw e;
   } catch (Exception e) {
    String lastErrorMessage = Nd4j.getNativeOps().lastErrorMessage();
    throw new ND4JIllegalStateException("Op name " + opName() + " - no output arrays were provided and calculateOutputShape failed to execute error message: " + lastErrorMessage, e);
   }
  }

  return shapeOverride;

 }
}
