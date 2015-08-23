package org.nd4j.bytebuddy.method.reference;

import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

/**
 * @author Adam Gibson
 */
public class LoadReferenceParamImplementation implements Implementation {
    private int offset = -1;

    public LoadReferenceParamImplementation(int offset) {
        this.offset = offset;
    }

    @Override
    public InstrumentedType prepare(InstrumentedType instrumentedType) {
        return instrumentedType;
    }

    @Override
    public ByteCodeAppender appender(Target implementationTarget) {
        return new LoadReferenceParam(offset);
    }
}
