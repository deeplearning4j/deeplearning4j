package org.nd4j.bytebuddy.method.integer;

import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

/**
 * @author Adam Gibson
 */
public class LoadIntParamImplementation implements Implementation {
    private int offset = -1;

    public LoadIntParamImplementation(int offset) {
        this.offset = offset;
    }

    @Override
    public InstrumentedType prepare(InstrumentedType instrumentedType) {
        return instrumentedType;
    }

    @Override
    public ByteCodeAppender appender(Target implementationTarget) {
        return new LoadIntParam(offset);
    }
}
