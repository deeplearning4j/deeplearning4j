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
package org.arbiter.optimize.parameter.continuous;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.arbiter.optimize.api.ParameterSpace;

import java.util.Collections;
import java.util.List;

public class ContinuousParameterSpace implements ParameterSpace<Double> {

    private RealDistribution distribution;
    private int index;

    public ContinuousParameterSpace(double min, double max){
        this(new UniformRealDistribution(min,max));
    }

    public ContinuousParameterSpace(RealDistribution distribution){
        this.distribution = distribution;
    }


    @Override
    public Double getValue(double[] input) {
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
    public String toString(){
        if(distribution instanceof UniformRealDistribution){
            return "ContinuousParameterSpace(min="+distribution.getSupportLowerBound()+",max="+distribution.getSupportUpperBound()+")";
        } else {
            return "ContinuousParameterSpace(" + distribution + ")";
        }
    }
}
