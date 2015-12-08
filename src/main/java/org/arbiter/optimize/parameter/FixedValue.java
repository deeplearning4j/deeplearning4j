package org.arbiter.optimize.parameter;

public class FixedValue<T> implements ParameterSpace<T> {
    private T value;

    public FixedValue(T value) {
        this.value = value;
    }

    @Override
    public T randomValue() {
        return value;
    }
}
