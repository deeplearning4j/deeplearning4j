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
