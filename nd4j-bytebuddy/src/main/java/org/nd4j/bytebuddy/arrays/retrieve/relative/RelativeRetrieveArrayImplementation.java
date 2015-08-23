package org.nd4j.bytebuddy.arrays.retrieve.relative;

import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

/**
 * @author Adam Gibson
 */
public class RelativeRetrieveArrayImplementation implements Implementation {
    private int index;

    public RelativeRetrieveArrayImplementation(int index) {
        this.index = index;
    }

    @Override
    public InstrumentedType prepare(InstrumentedType instrumentedType) {
        return instrumentedType;
    }

    @Override
    public ByteCodeAppender appender(Target implementationTarget) {
        return new RelativeRetrieveArrayValueAppender(index);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof RelativeRetrieveArrayImplementation)) return false;

        RelativeRetrieveArrayImplementation that = (RelativeRetrieveArrayImplementation) o;

        return index == that.index;

    }

    @Override
    public int hashCode() {
        return index;
    }
}
