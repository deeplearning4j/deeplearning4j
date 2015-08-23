package org.nd4j.bytebuddy.storeint;

import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

/**
 * @author Adam Gibson
 */
public class StoreIntImplementation implements Implementation {
    private int storeId = -1;

    public StoreIntImplementation(int storeId) {
        this.storeId = storeId;
    }

    @Override
    public InstrumentedType prepare(InstrumentedType instrumentedType) {
        return instrumentedType;
    }

    @Override
    public ByteCodeAppender appender(Target implementationTarget) {
        return new StoreInt(storeId);
    }
}
