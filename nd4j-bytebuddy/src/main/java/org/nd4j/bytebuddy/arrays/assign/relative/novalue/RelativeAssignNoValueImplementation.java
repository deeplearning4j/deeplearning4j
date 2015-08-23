package org.nd4j.bytebuddy.arrays.assign.relative.novalue;

import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

/**
 * @author Adam Gibson
 */
public class RelativeAssignNoValueImplementation implements Implementation {
    private int index;

    public RelativeAssignNoValueImplementation(int index) {
        this.index = index;
    }

    @Override
    public InstrumentedType prepare(InstrumentedType instrumentedType) {
        return instrumentedType;
    }

    @Override
    public ByteCodeAppender appender(Target implementationTarget) {
        return new RelativeAssignNoValueArrayValueAppender(index);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof RelativeAssignNoValueImplementation)) return false;

        RelativeAssignNoValueImplementation that = (RelativeAssignNoValueImplementation) o;

        return index == that.index;

    }

    @Override
    public int hashCode() {
        return index;
    }
}
