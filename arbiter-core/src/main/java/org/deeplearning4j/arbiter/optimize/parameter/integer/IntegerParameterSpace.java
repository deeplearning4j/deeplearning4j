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
package org.deeplearning4j.arbiter.optimize.parameter.integer;

import lombok.EqualsAndHashCode;
import org.apache.commons.math3.distribution.IntegerDistribution;
import org.apache.commons.math3.distribution.UniformIntegerDistribution;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.List;

/**
 * IntegerParameterSpace is a {@code ParameterSpace<Integer>}; i.e., defines an ordered space of integers between
 * some minimum and maximum value
 *
 * @author Alex Black
 */
@JsonIgnoreProperties("index")
@EqualsAndHashCode
public class IntegerParameterSpace implements ParameterSpace<Integer> {

    private IntegerDistribution distribution;
    private int index = -1;

    /**
     * Create an IntegerParameterSpace with a uniform distribution between the specified min/max (inclusive)
     *
     * @param min Min value, inclusive
     * @param max Max value, inclusive
     */
    public IntegerParameterSpace(int min, int max) {
        this(new UniformIntegerDistribution(min, max));
    }

    /**
     * Crate an IntegerParametSpace from the given IntegerDistribution
     *
     * @param distribution Distribution to use
     */
    public IntegerParameterSpace(@JsonProperty("distribution") IntegerDistribution distribution) {
        this.distribution = distribution;
    }

    public int getMin() {
        return distribution.getSupportLowerBound();
    }

    public int getMax() {
        return distribution.getSupportUpperBound();
    }

    @Override
    public Integer getValue(double[] input) {
        if (index == -1){
            throw new IllegalStateException("Cannot get value: ParameterSpace index has not been set");
        }
        return distribution.inverseCumulativeProbability(input[index]);
    }

    @Override
    public int numParameters() {
        return 1;
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return Collections.singletonList((ParameterSpace) this);
    }

    @Override
    public boolean isLeaf() {
        return true;
    }

    @Override
    public void setIndices(int... indices) {
        if (indices == null || indices.length != 1) throw new IllegalArgumentException("Invalid index");
        this.index = indices[0];
    }

    @Override
    public String toString() {
        if (distribution instanceof UniformIntegerDistribution) {
            return "IntegerParameterSpace(min=" + distribution.getSupportLowerBound() + ",max=" + distribution.getSupportUpperBound() + ")";
        } else {
            return "IntegerParameterSpace(" + distribution + ")";
        }
    }

}
