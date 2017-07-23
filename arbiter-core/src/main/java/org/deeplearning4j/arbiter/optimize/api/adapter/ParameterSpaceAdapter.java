package org.deeplearning4j.arbiter.optimize.api.adapter;

import lombok.AllArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;

import java.util.List;
import java.util.Map;

/**
 * Created by Alex on 23/07/2017.
 */
@AllArgsConstructor
public abstract class ParameterSpaceAdapter<F,T> implements ParameterSpace<T> {


    protected abstract T convertValue(F from);

    protected abstract ParameterSpace<F> underlying();


    @Override
    public T getValue(double[] parameterValues) {
        return convertValue(underlying().getValue(parameterValues));
    }

    @Override
    public int numParameters() {
        return underlying().numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return underlying().collectLeaves();
    }

    @Override
    public Map<String, ParameterSpace> getNestedSpaces() {
        return underlying().getNestedSpaces();
    }

    @Override
    public boolean isLeaf() {
        return underlying().isLeaf();
    }

    @Override
    public void setIndices(int... indices) {
        underlying().setIndices(indices);
    }

    @Override
    public String toString() {
        return underlying().toString();
    }
}
