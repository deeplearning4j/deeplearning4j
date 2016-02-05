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
package org.arbiter.optimize.parameter.integer;

import org.apache.commons.math3.distribution.IntegerDistribution;
import org.apache.commons.math3.distribution.UniformIntegerDistribution;
import org.arbiter.optimize.api.ParameterSpace;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class IntegerParameterSpace implements ParameterSpace<Integer> {

    private IntegerDistribution distribution;
    private int index;

    /**
     * @param min Min value, inclusive
     * @param max Max value, inclusive
     */
    public IntegerParameterSpace(int min, int max){
        this(new UniformIntegerDistribution(min,max));
    }

    public IntegerParameterSpace(IntegerDistribution distribution){
        this.distribution = distribution;
    }

    @Override
    public Integer getValue(double[] input) {
        if(input == null || input.length != 1)  throw new IllegalArgumentException("Invalid input: must be length 1");
        if(input[0] < 0.0 || input[0] > 1.0) throw new IllegalArgumentException("Invalid input: input must be in range 0 to 1 inclusive");
        return distribution.inverseCumulativeProbability(input[0]);
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
        if(indices == null || indices.length != 1) throw new IllegalArgumentException("Invalid index");
        this.index = indices[0];
    }

    @Override
    public String toString() {
        if (distribution instanceof UniformIntegerDistribution) {
            return "IntegerParameterSpace(min="+distribution.getSupportLowerBound() + ",max="+distribution.getSupportUpperBound()+")";
        } else {
            return "IntegerParameterSpace("+distribution+")";
        }
    }
}
