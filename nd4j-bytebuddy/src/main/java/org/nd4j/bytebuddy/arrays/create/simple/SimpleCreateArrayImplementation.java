package org.nd4j.bytebuddy.arrays.create.simple;

import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

/**
 * @author Adam Gibson
 */
public class SimpleCreateArrayImplementation implements Implementation {
    private int length = -1;

    public SimpleCreateArrayImplementation(int length) {
        this.length = length;
    }

    @Override
    public InstrumentedType prepare(InstrumentedType instrumentedType) {
        return instrumentedType;
    }

    @Override
    public ByteCodeAppender appender(Target implementationTarget) {
        return new SimpleCreateArrayByteCodeAppender(length);
    }
}
