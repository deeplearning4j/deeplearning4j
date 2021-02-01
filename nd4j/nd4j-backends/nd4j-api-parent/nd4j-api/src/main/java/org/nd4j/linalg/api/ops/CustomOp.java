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

package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;

import java.util.List;

/**
 * This interface describe "custom operations.
 * Originally these operations are designed for SameDiff, and execution within graph,
 * but we also want to provide option to use them with regular ND4J methods via NativeOpExecutioner
 *
 * @author raver119@gmail.com
 */
public interface CustomOp {
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

    void addIArgument(int... arg);

    void addIArgument(long... arg);

    void addBArgument(boolean... arg);

    void addDArgument(DataType... arg);

    void removeIArgument(Integer arg);

    Boolean getBArgument(int index);

    Long getIArgument(int index);

    int numIArguments();

    void removeTArgument(Double arg);

    Double getTArgument(int index);

    int numTArguments();

    int numBArguments();

    int numDArguments();

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
}
