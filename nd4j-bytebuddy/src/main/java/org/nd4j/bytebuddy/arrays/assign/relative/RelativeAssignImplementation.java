package org.nd4j.bytebuddy.arrays.assign.relative;

import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

/**
 * @author Adam Gibson
 */
public class RelativeAssignImplementation implements Implementation {
    private int index,val;

    public RelativeAssignImplementation(int index, int val) {
        this.index = index;
        this.val = val;
    }

    @Override
    public InstrumentedType prepare(InstrumentedType instrumentedType) {
        return instrumentedType;
    }

    @Override
    public ByteCodeAppender appender(Target implementationTarget) {
        return new RelativeAssignArrayValueAppender(index,val);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof RelativeAssignImplementation)) return false;

        RelativeAssignImplementation that = (RelativeAssignImplementation) o;

        if (index != that.index) return false;
        return val == that.val;

    }

    @Override
    public int hashCode() {
        int result = index;
        result = 31 * result + val;
        return result;
    }
}
