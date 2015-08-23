package org.nd4j.bytebuddy.storeref;

import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

/**
 * @author Adam Gibson
 */
public class StoreImplementation implements Implementation {
    private int storeId = -1;

    public StoreImplementation(int storeId) {
        this.storeId = storeId;
    }

    @Override
    public InstrumentedType prepare(InstrumentedType instrumentedType) {
        return instrumentedType;
    }

    @Override
    public ByteCodeAppender appender(Target implementationTarget) {
        return new StoreRef(storeId);
    }
}
