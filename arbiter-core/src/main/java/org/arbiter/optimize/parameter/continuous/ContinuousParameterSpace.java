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
import org.arbiter.optimize.parameter.ParameterSpace;

public class ContinuousParameterSpace implements ParameterSpace<Double> {

    private RealDistribution distribution;

    public ContinuousParameterSpace(double min, double max){
        this(new UniformRealDistribution(min,max));
    }

    public ContinuousParameterSpace(RealDistribution distribution){
        this.distribution = distribution;
    }

    @Override
    public Double randomValue() {
        return distribution.sample();
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
