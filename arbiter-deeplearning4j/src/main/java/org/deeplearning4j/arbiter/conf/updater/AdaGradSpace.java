package org.deeplearning4j.arbiter.conf.updater;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.IUpdater;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class AdaGradSpace implements ParameterSpace<IUpdater> {

    private ParameterSpace<Double> learningRate;

    @Getter @Setter
    private int[] indices;

    public AdaGradSpace(@NonNull ParameterSpace<Double> learningRate){
        this.learningRate = learningRate;
    }

    @Override
    public IUpdater getValue(double[] parameterValues) {
        return new AdaGrad(learningRate.getValue(parameterValues));
    }

    @Override
    public int numParameters() {
        return learningRate.numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return learningRate.collectLeaves();
    }

    @Override
    public Map<String, ParameterSpace> getNestedSpaces() {
        return Collections.<String,ParameterSpace>singletonMap("learningRate", learningRate);
    }

    @Override
    public boolean isLeaf() {
        return false;
    }
}
