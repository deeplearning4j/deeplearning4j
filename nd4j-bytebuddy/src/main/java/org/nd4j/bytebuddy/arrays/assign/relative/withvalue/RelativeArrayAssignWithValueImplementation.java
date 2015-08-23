package org.nd4j.bytebuddy.arrays.assign.relative.withvalue;

import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

/**
 * @author Adam Gibson
 */
public class RelativeArrayAssignWithValueImplementation implements Implementation {
    private int index,val;

    public RelativeArrayAssignWithValueImplementation(int index, int val) {
        this.index = index;
        this.val = val;
    }

    @Override
    public InstrumentedType prepare(InstrumentedType instrumentedType) {
        return instrumentedType;
    }

    @Override
    public ByteCodeAppender appender(Target implementationTarget) {
        return new RelativeArrayAssignWithValueValueAppender(index,val);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof RelativeArrayAssignWithValueImplementation)) return false;

        RelativeArrayAssignWithValueImplementation that = (RelativeArrayAssignWithValueImplementation) o;

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
