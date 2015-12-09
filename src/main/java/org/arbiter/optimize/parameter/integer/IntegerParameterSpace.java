package org.arbiter.optimize.parameter.integer;

import org.apache.commons.math3.distribution.IntegerDistribution;
import org.apache.commons.math3.distribution.UniformIntegerDistribution;
import org.arbiter.optimize.parameter.ParameterSpace;

public class IntegerParameterSpace implements ParameterSpace<Integer> {

    private IntegerDistribution distribution;

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
}
