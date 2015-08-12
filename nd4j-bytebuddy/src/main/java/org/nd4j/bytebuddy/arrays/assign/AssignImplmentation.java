package org.nd4j.bytebuddy.arrays.assign;

import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

/**
 * The actual implementation
 * of assigning a value to an array
 * @author Adam Gibson
 */
public class AssignImplmentation implements Implementation {
    private int index,val;

    /**
     *
     * @param index the index in the array to assign the value
     * @param val the value to assign
     */
    public AssignImplmentation(int index, int val) {
        this.index = index;
        this.val = val;
    }

    @Override
    public InstrumentedType prepare(InstrumentedType instrumentedType) {
        return instrumentedType;
    }

    @Override
    public ByteCodeAppender appender(Target implementationTarget) {
        return new AssignArrayValueAppender(index,val);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof AssignImplmentation)) return false;

        AssignImplmentation that = (AssignImplmentation) o;

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
