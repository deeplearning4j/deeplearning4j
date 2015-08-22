package org.nd4j.bytebuddy.createint;

import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

/**
 * Instantiate an int with the given value
 * as a variable.
 * An example is:
 * int i = 5;
 * @author Adam Gibson
 */
public class StoreIntImplementation implements Implementation {

   private int idx = -1;

    /**
     * Specify the variable index
     * @param idx
     */
    public StoreIntImplementation(int idx) {
        this.idx = idx;
    }

    @Override
    public InstrumentedType prepare(InstrumentedType instrumentedType) {
        return instrumentedType;
    }

    @Override
    public ByteCodeAppender appender(Target implementationTarget) {
        return new StoreInt(idx);
    }


}
