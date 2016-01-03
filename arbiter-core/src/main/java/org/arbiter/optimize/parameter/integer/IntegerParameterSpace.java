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
import org.arbiter.optimize.parameter.ParameterSpace;

public class IntegerParameterSpace implements ParameterSpace<Integer> {

    private IntegerDistribution distribution;

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
    public Integer randomValue() {
        return distribution.sample();
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
