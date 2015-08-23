package org.nd4j.bytebuddy.method.args;

import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

/**
 * @author Adam Gibson
 */
public class LoadArgsImplementation implements Implementation {
    private boolean loadThis = false;

    public LoadArgsImplementation(boolean loadThis) {
        this.loadThis = loadThis;
    }

    public LoadArgsImplementation() {
        this(false);
    }

    @Override
    public InstrumentedType prepare(InstrumentedType instrumentedType) {
        return instrumentedType;
    }

    @Override
    public ByteCodeAppender appender(Target implementationTarget) {
        return new LoadArgsAppender(loadThis);
    }
}
