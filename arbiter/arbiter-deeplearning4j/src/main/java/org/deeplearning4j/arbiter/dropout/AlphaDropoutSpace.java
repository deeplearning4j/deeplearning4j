package org.deeplearning4j.arbiter.dropout;

import lombok.AllArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.nn.conf.dropout.AlphaDropout;
import org.deeplearning4j.nn.conf.dropout.IDropout;

import java.util.Collections;
import java.util.List;
import java.util.Map;

@AllArgsConstructor
public class AlphaDropoutSpace implements ParameterSpace<IDropout> {

    private ParameterSpace<Double> dropout;

    public AlphaDropoutSpace(double activationRetainProbability){
        this(new FixedValue<>(activationRetainProbability));
    }

    @Override
    public IDropout getValue(double[] parameterValues) {
        return new AlphaDropout(dropout.getValue(parameterValues));
    }

    @Override
    public int numParameters() {
        return dropout.numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return Collections.<ParameterSpace>singletonList(dropout);
    }

    @Override
    public Map<String, ParameterSpace> getNestedSpaces() {
        return Collections.<String,ParameterSpace>singletonMap("dropout", dropout);
    }

    @Override
    public boolean isLeaf() {
        return false;
    }

    @Override
    public void setIndices(int... indices) {
        dropout.setIndices(indices);
    }
}
