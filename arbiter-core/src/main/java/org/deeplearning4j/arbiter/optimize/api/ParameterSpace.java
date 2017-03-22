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

import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.util.List;

/**
 * ParameterSpace: defines the acceptable ranges of values a given parameter may take.
 * Note that parameter spaces can be simple (like {@code ParameterSpace<Double>}) or complicated, including
 * multiple nested ParameterSpaces
 *
 * @author Alex Black
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonSubTypes(value={
        @JsonSubTypes.Type(value = ContinuousParameterSpace.class, name = "ContinuousParameterSpace"),
        @JsonSubTypes.Type(value = DiscreteParameterSpace.class, name = "DiscreteParameterSpace"),
        @JsonSubTypes.Type(value = IntegerParameterSpace.class, name = "IntegerParameterSpace"),
        @JsonSubTypes.Type(value = FixedValue.class, name = "FixedValue")
})
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "type")
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
     * Collect a list of parameters, recursively.
     */
    List<ParameterSpace> collectLeaves();

    /**
     * Is this ParameterSpace a leaf? (i.e., does it contain other ParameterSpace values?)
     */
    @JsonIgnore
    boolean isLeaf();

    /**
     * For leaf ParameterSpaces: set the indices of the leaf ParameterSpace.
     * Expects input of length {@link #numParameters()}. Throws exception if {@link #isLeaf()} is false.
     *
     * @param indices
     */
    void setIndices(int... indices);

}
