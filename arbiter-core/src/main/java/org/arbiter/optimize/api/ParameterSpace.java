/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.arbiter.optimize.api;

import java.util.List;

/**ParameterSpace: defines the acceptable ranges of values a given parameter may take
 */
public interface ParameterSpace<P> {

//    P randomValue();

//    /**Map an input value (0 to 1) to an output value. Can be used to generate random values, for example.
//     * @param input Input value, in range [0,1].
//     * @return An output value
//     */
//    P getValue(double input);

    /** Generate a candidate given a set of values. These values are then mapped to a specific candidate, using some
     * mapping function (such as the prior probability distribution)
     * @param parameterValues A set of values, each in the range [0,1], of length {@link #numParameters()}
     */
    P getValue(double[] parameterValues);

    /** Get the total number of parameters (hyperparameters) to be optimized. This includes optional parameters from
     * different parameter subpaces. (Thus, not every parameter may be used in every candidate)
     * @return Number of hyperparameters to be optimized
     */
    int numParameters();

    /** Collect a list of parameters, recursively. */
    List<ParameterSpace> collectLeaves();

    /** Is this ParameterSpace a leaf? (i.e., does it contain other ParameterSpace values?) */
    boolean isLeaf();

    /** For leaf ParameterSpaces: set the indices of the leaf ParameterSpace.
     * Expects input of length {@link #numParameters()}. Throws exception if {@link #isLeaf()} is false.
     * @param indices
     */
    void setIndices(int... indices);

}
