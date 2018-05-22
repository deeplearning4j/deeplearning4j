package org.deeplearning4j.arbiter.dropout;

import lombok.AllArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.nn.conf.dropout.GaussianNoise;
import org.deeplearning4j.nn.conf.dropout.IDropout;

import java.util.Collections;
import java.util.List;
import java.util.Map;

@AllArgsConstructor
public class GaussianNoiseSpace implements ParameterSpace<IDropout> {

    private ParameterSpace<Double> stddev;

    public GaussianNoiseSpace(double stddev){
        this(new FixedValue<>(stddev));
    }

    @Override
    public IDropout getValue(double[] parameterValues) {
        return new GaussianNoise(stddev.getValue(parameterValues));
    }

    @Override
    public int numParameters() {
        return stddev.numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return Collections.<ParameterSpace>singletonList(stddev);
    }

    @Override
    public Map<String, ParameterSpace> getNestedSpaces() {
        return Collections.<String,ParameterSpace>singletonMap("stddev", stddev);
    }

    @Override
    public boolean isLeaf() {
        return false;
    }

    @Override
    public void setIndices(int... indices) {
        stddev.setIndices(indices);
    }
}
