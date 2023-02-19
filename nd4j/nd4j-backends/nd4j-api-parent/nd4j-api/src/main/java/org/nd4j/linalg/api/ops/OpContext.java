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

import java.util.List;

public interface OpContext extends AutoCloseable {

    /**
     * Returns the id of the op context (mainly used for tracking purposes)
     * @return
     */
    long id();

    /**
     * Returns true if the op context
     * currently contains a valid data type arguments pointer
     * @return
     */

    boolean hasCachedDArgs();
    /**
     * Returns true if the op context
     * currently contains a valid double arguments pointer
     * @return
     */

    boolean hasCachedTArgs();
    /**
     * Returns true if the op context
     * currently contains a valid boolean arguments pointer
     * @return
     */

    boolean hasCachedBArgs();
    /**
     * Returns true if the op context
     * currently contains a valid int arguments pointer
     * @return
     */

    boolean hasCachedIArgs();

    /**
     * This method sets an tArgument at an index
     * @param index the index to set
     * @param value the value to set
     */
    void setDArgAt(int index,DataType value);


    /**
     * This method sets an bArgument at an index
     * @param index the index to set
     * @param value the value to set
     */
    void setBArgAt(int index,boolean value);


    /**
     * This method sets an tArgument at an index
     * @param index the index to set
     * @param value the value to set
     */
    void setTArgAt(int index,double value);

    /**
     * This method sets an iArgument at an index
     * @param index the index to set
     * @param value the value to set
     */
    void setIArgAt(int index,long value);
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

    List<Long> getIArguments();

    int numIArguments();

    /**
     * This method sets floating point arguments required for operation
     * @param arguments
     */
    void setTArguments(double... arguments);

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
     * This method sets data type arguments required for operation
     * @param arguments
     */
    void setDArguments(DataType... arguments);


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
     * This method sets data type arguments required for operation
     *
     * @param arguments
     * @param length
     */
    void setBArguments(Pointer arguments, int length);

    /**
     * This method sets boolean arguments required for operation
     * @param arguments
     */
    void setBArguments(boolean... arguments);



    List<Boolean> getBArguments();
    int numBArguments();

    /**
     * This method sets root-level seed for rng
     * @param seed
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
     * This method returns pointer to context, to be used during native op execution
     * @return
     */
    Pointer contextPointer();

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
     * This methos allows to disape outputs validation via shape function
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
