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
