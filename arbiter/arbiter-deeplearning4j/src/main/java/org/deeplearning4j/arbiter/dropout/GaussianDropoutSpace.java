package org.deeplearning4j.arbiter.dropout;

import lombok.AllArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.dropout.GaussianDropout;
import org.deeplearning4j.nn.conf.dropout.IDropout;

import java.util.Collections;
import java.util.List;
import java.util.Map;

@AllArgsConstructor
public class GaussianDropoutSpace implements ParameterSpace<IDropout> {

    private ParameterSpace<Double> rate;

    public GaussianDropoutSpace(double rate){
        this(new FixedValue<>(rate));
    }

    @Override
    public IDropout getValue(double[] parameterValues) {
        return new GaussianDropout(rate.getValue(parameterValues));
    }

    @Override
    public int numParameters() {
        return rate.numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return Collections.<ParameterSpace>singletonList(rate);
    }

    @Override
    public Map<String, ParameterSpace> getNestedSpaces() {
        return Collections.<String,ParameterSpace>singletonMap("rate", rate);
    }

    @Override
    public boolean isLeaf() {
        return false;
    }

    @Override
    public void setIndices(int... indices) {
        rate.setIndices(indices);
    }
}
