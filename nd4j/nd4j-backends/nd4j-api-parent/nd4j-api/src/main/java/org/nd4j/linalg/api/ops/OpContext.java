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

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.common.primitives.Pair;
import org.nd4j.nativeblas.OpaqueContext;

import java.util.List;

public interface OpContext extends AutoCloseable {

    /**
     * Returns the id of the op context (mainly used for tracking purposes)
     * @return
     */
    long id();


    /**
     * Copies arguments from the given CustomOp
     * @param customOp CustomOp to copy arguments from
     */
    void setArgsFrom(CustomOp customOp);

    /**
     * This method sets integer arguments required for operation
     *
     * @param arguments
     * @param length
     */
    void setIArguments(Pointer arguments, int length);
    /**
     * This method sets integer arguments required for operation
     * @param arguments
     */
    void setIArguments(long... arguments);

    void setIArguments(List<Long> iArguments);

    List<Long> getIArguments();

    int numIArguments();


    /**
     * This method returns integer argument by index
     * from the c++ level. This is mainly used for testing.
     * @param index index of the argument
     * @return argument
     */
    int iArgumentAtNative(int index);

    /**
     * This method returns integer argument by index
     * from the c++ level. This is mainly used for testing.

     * @return
     */
    int numIArgumentsNative();

    /**
     * This method sets floating point arguments required for operation
     * @param arguments
     */
    void setTArguments(double... arguments);


    void setTArguments(List<Double> tArguments);

    /**
     * This method sets floating point arguments required for operation
     *
     * @param arguments
     * @param length
     */
    void setTArguments(Pointer arguments, int length);

    List<Double> getTArguments();
    int numTArguments();


    /**
     * This method returns floating point argument by index
     * from the c++ level. This is mainly used for testing.
     * @param index index of the argument
     * @return argument
     */
    Double tArgumentNative(int index);

    /**
     * This method returns the number of floating point arguments
     * from the c++ level. This is mainly used for testing.
     * @return
     */
    int numTArgumentsNative();

    /**
     * This method sets data type arguments required for operation
     * @param arguments
     */
    void setDArguments(DataType... arguments);

    void setDArguments(List<DataType> arguments);

    /**
     * This method sets data type arguments required for operation
     *
     * @param arguments
     * @param length
     */
    void setDArguments(Pointer arguments, int length);

    List<DataType> getDArguments();
    int numDArguments();

    /**
     * Returns the data type
     * from the underlying c++.
     * Mainly used for testing.
     * @param index
     * @return
     */
    DataType dataTypeNativeAt(int index);

    /**
     * This method returns number of data type arguments
     * from c++. This is mainly used for testing.
     * @return
     */
    int numDNative();

    /**
     * This method returns number of intermediate results
     * @return
     */
    int numIntermediateResults();

    /**
     * This method sets intermediate result for future op call
     * @param index
     * @param arr
     */
    void setIntermediateResult(int index,INDArray arr);

    /**
     * This method returns intermediate result by index
     * @param index
     * @return
     */
    INDArray getIntermediateResult(int index);

    /**
     * This method adds intermediate result for future op call
     * @param arr
     */
    void addIntermediateResult(INDArray arr);

    /**
     * This method sets data type arguments required for operation
     *
     * @param arguments
     * @param length
     */
    void setBArguments(Pointer arguments, int length);


    void setBArguments(List<Boolean> arguments);


    /**
     * This method sets boolean arguments required for operation
     * @param arguments
     */
    void setBArguments(boolean... arguments);



    List<Boolean> getBArguments();
    int numBArguments();

    /**
     * This method returns boolean argument by index
     * from the c++ level. This is mainly used for testing.
     * @param index index of the argument
     * @return
     */
    boolean bArgumentAtNative(int index);

    /**
     * This method returns number of boolean arguments
     * from the c++ level. This is mainly used for testing.
     * @return
     */
    int numBArgumentsNative();


    /**
     * This method sets root-level seed for rng
     * @param rootState
     * @param nodeState
     */
    void setRngStates(long rootState, long nodeState);

    /**
     * This method returns RNG states, root first node second
     * @return
     */
    Pair<Long, Long> getRngStates();

    /**
     * This method adds INDArray as input argument for future op call
     *
     * @param index
     * @param array
     */
    void setInputArray(int index, INDArray array);

    /**
     * This method sets provided arrays as input arrays
     * @param arrays
     */
    void setInputArrays(List<INDArray> arrays);

    /**
     * This method sets provided arrays as input arrays
     * @param arrays
     */
    void setInputArrays(INDArray... arrays);

    /**
     * This method returns List of input arrays defined within this context
     * @return
     */
    List<INDArray> getInputArrays();

    int numInputArguments();

    INDArray getInputArray(int idx);

    /**
     * This method returns input array by index
     * from the c++ level. This is mainly used for testing.
     * @param idx index of the argument
     * @return input array
     */
    INDArray getInputArrayNative(int idx);

    /**
     * This method returns number of input arguments
     * from the c++ level. This is mainly used for testing.
     * @return
     */
    int numInputsNative();

    /**
     * This method adds INDArray as output for future op call
     * @param index
     * @param array
     */
    void setOutputArray(int index, INDArray array);

    /**
     * This method sets provided arrays as output arrays
     * @param arrays
     */
    void setOutputArrays(List<INDArray> arrays);

    /**
     * This method sets provided arrays as output arrays
     * @param arrays
     */
    void setOutputArrays(INDArray... arrays);

    /**
     * This method returns List of output arrays defined within this context
     * @return
     */
    List<INDArray> getOutputArrays();

    INDArray getOutputArray(int i);

    int numOutputArguments();


    /**
     * This method returns output array by index
     * from the c++ level. This is mainly used for testing.
     * @param idx
     * @return
     */
    INDArray getOutputArrayNative(int idx);

    /**
     * This method returns number of outputs
     * from the c++ level. This is mainly used for testing.
     * @return output array
     */
    int numOutArgumentsNative();

    /**
     * This method returns pointer to context, to be used during native op execution
     *
     * @return
     */
    OpaqueContext contextPointer();

    /**
     * This method allows to set op as inplace
     * @param reallyInplace
     */
    void markInplace(boolean reallyInplace);

    /**
     * This method allows to enable/disable use of platform helpers within ops. I.e. mkldnn or cuDNN.
     * PLEASE NOTE: default value is True
     *
     * @param reallyAllow
     */
    void allowHelpers(boolean reallyAllow);

    /**
     * This method allows to display outputs validation via shape function
     * @param reallyOverride
     */
    void shapeFunctionOverride(boolean reallyOverride);

    /**
     * This method returns current execution mode for Context
     * @return
     */
    ExecutionMode getExecutionMode();

    /**
     * This method allows to set certain execution mode
     *
     * @param mode
     */
    void setExecutionMode(ExecutionMode mode);

    /**
     * This method removes all in/out arrays from this OpContext
     */
    void purge();


    /**
     * set context arguments
     * @param inputArrs
     * @param iArgs
     * @param dArgs
     * @param tArgs
     * @param bArgs
     */
    void setArgs(INDArray[] inputArrs, long[] iArgs, DataType[] dArgs, double[] tArgs, boolean[] bArgs);

    /**
     * Transfers double arguments in java to
     * c++
     */
    void transferTArgs();

    /**
     * Transfers int arguments in java to c++
     */
    void transferIArgs();

    /**
     * Transfers boolean arguments in java to c++
     */
    void transferBArgs();

    /**
     * Transfers data type arguments to c++
     */
    void transferDArgs();

}
