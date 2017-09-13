package org.deeplearning4j.arbiter.conf.dropout;

import lombok.AllArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.AbstractParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.dropout.IDropout;

import java.util.List;

@AllArgsConstructor
public class DropoutSpace extends AbstractParameterSpace<IDropout> {

    private ParameterSpace<Double> dropout;

    @Override
    public Dropout getValue(double[] parameterValues) {
        return new Dropout(dropout.getValue(parameterValues));
    }

    @Override
    public int numParameters() {
        return dropout.numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return dropout.collectLeaves();
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
