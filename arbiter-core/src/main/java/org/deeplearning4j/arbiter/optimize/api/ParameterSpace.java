/*-
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
package org.deeplearning4j.arbiter.optimize.api;

import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.util.List;
import java.util.Map;

/**
 * ParameterSpace: defines the acceptable ranges of values a given parameter may take.
 * Note that parameter spaces can be simple (like {@code ParameterSpace<Double>}) or complicated, including
 * multiple nested ParameterSpaces
 *
 * @author Alex Black
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface ParameterSpace<P> {

    /**
     * Generate a candidate given a set of values. These values are then mapped to a specific candidate, using some
     * mapping function (such as the prior probability distribution)
     *
     * @param parameterValues A set of values, each in the range [0,1], of length {@link #numParameters()}
     */
    P getValue(double[] parameterValues);

    /**
     * Get the total number of parameters (hyperparameters) to be optimized. This includes optional parameters from
     * different parameter subpaces. (Thus, not every parameter may be used in every candidate)
     *
     * @return Number of hyperparameters to be optimized
     */
    int numParameters();

    /**
     * Collect a list of parameters, recursively. Note that leaf parameters are parameters that do not have any
     * nested parameter spaces
     */
    List<ParameterSpace> collectLeaves();

    /**
     * Get a list of nested parameter spaces by name. Note that the returned parameter spaces may in turn have further
     * nested parameter spaces. The map should be empty for leaf parameter spaces
     *
     * @return  A map of nested parameter spaces
     */
    Map<String, ParameterSpace> getNestedSpaces();

    /**
     * Is this ParameterSpace a leaf? (i.e., does it contain other ParameterSpaces internally?)
     */
    @JsonIgnore
    boolean isLeaf();

    /**
     * For leaf ParameterSpaces: set the indices of the leaf ParameterSpace.
     * Expects input of length {@link #numParameters()}. Throws exception if {@link #isLeaf()} is false.
     *
     * @param indices Indices to set. Length should equal {@link #numParameters()}
     */
    void setIndices(int... indices);

}
