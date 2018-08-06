package org.deeplearning4j.arbiter.optimize.parameter;

import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * BooleanParameterSpace is a {@code ParameterSpace<Boolean>}; Defines {True, False} as a parameter space
 * If argument to setValue is less than or equal to 0.5 it will return True else False
 *
 * @author susaneraly
 */
public class BooleanSpace implements ParameterSpace<Boolean> {
    private int index = -1;

    @Override
    public Boolean getValue(double[] input) {
        if (index == -1) {
            throw new IllegalStateException("Cannot get value: ParameterSpace index has not been set");
        }
        if (input[index] <= 0.5) return Boolean.TRUE;
        else return Boolean.FALSE;
    }

    @Override
    public int numParameters() {
        return 1;
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return Collections.singletonList((ParameterSpace) this);
    }

    @Override
    public Map<String, ParameterSpace> getNestedSpaces() {
        return Collections.emptyMap();
    }

    @Override
    public boolean isLeaf() {
        return true;
    }

    @Override
    public void setIndices(int... indices) {
        if (indices == null || indices.length != 1)
            throw new IllegalArgumentException("Invalid index");
        this.index = indices[0];
    }

    @Override
    public String toString() {
        return "BooleanSpace()";
    }
}
