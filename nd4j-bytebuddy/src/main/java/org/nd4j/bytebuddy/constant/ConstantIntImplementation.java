package org.nd4j.bytebuddy.constant;

import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

/**
 * An implementation that puts
 * an int on the stack.
 *
 * @author Adam Gibson
 */
public class ConstantIntImplementation implements Implementation {
    private int val = -1;

    public ConstantIntImplementation(int val) {
        this.val = val;
    }

    @Override
    public InstrumentedType prepare(InstrumentedType instrumentedType) {
        return instrumentedType;
    }

    @Override
    public ByteCodeAppender appender(Target implementationTarget) {
        return new IntConstantByteCodeAppender(val);
    }
}
