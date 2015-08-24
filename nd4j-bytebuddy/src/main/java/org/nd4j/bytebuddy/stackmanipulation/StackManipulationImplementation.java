package org.nd4j.bytebuddy.stackmanipulation;

import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.StackManipulation;

/**
 * Stack manipulation implementation:
 * Allows top level specification of stack manipulations
 * for more fine grained control.
 *
 * @author Adam Gibson
 */
public class StackManipulationImplementation implements Implementation {
    private StackManipulation stackManipulation;

    public StackManipulationImplementation(StackManipulation stackManipulation) {
        this.stackManipulation = stackManipulation;
    }

    @Override
    public InstrumentedType prepare(InstrumentedType instrumentedType) {
        return instrumentedType;
    }

    @Override
    public ByteCodeAppender appender(Target implementationTarget) {
        return new StackManipulationByteCodeAppender(stackManipulation);
    }
}
