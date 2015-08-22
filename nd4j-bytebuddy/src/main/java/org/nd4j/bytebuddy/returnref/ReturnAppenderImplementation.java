package org.nd4j.bytebuddy.returnref;

import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

/**
 * @author Adam Gibson
 */
public class ReturnAppenderImplementation implements Implementation  {
    private ReturnAppender.ReturnType returnType;

    public ReturnAppenderImplementation(ReturnAppender.ReturnType returnType) {
        this.returnType = returnType;
    }

    @Override
    public InstrumentedType prepare(InstrumentedType instrumentedType) {
        return instrumentedType;
    }

    @Override
    public ByteCodeAppender appender(Target implementationTarget) {
        return new ReturnAppender(returnType);
    }
}
