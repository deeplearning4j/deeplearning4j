package org.arbiter.optimize.parameter;

import java.util.*;

public class DiscreteParameterSpace<P> implements ParameterSpace<P> {

    //TODO add distribution
    private List<P> values;
    private Random random = new Random();;  //TODO handling of seeds in global config

    public DiscreteParameterSpace(P... values){
        this.values = Arrays.asList(values);
    }

    public DiscreteParameterSpace(Collection<P> values){
        this.values = new ArrayList<>(values);
    }

    @Override
    public P randomValue() {
        int randomIdx = random.nextInt(values.size());
        return values.get(randomIdx);
    }
}
