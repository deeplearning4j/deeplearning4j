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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
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
  * @return Output array shapes
  */
 List<LongShapeDescriptor> calculateOutputShape();

 /**
  * Calculate the output shape for this op
  * @return Output array shapes
  */
 List<LongShapeDescriptor> calculateOutputShape(OpContext opContext);

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
  * Initialize the output arrays, if required.
  * @return True if the output arrays were initialized (and hence should be calculated), false otherwise
  */
 default boolean initializeOutputs(OpContext ctx) {
  boolean shapeOverride = false;
  if (numOutputArguments() == 0 && !isInplaceCall()) {
   try {
    val list = Nd4j.getExecutioner().calculateOutputShape(this,ctx);
    if (list.isEmpty())
     throw new ND4JIllegalStateException("Op name " + opName() + " failed to calculate output shape and data types.");

    for (LongShapeDescriptor shape : list) {
     INDArray newOut = Nd4j.create(shape, false);
     addOutputArgument(newOut);
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
