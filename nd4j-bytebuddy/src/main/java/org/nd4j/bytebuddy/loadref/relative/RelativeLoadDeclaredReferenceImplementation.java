package org.nd4j.bytebuddy.loadref.relative;

import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

/**
 * Load a reference from the stack
 * that was not apart of a method header.
 *
 * @author Adam Gibson
 */
public class RelativeLoadDeclaredReferenceImplementation implements Implementation {
    private int id = -1;

    public RelativeLoadDeclaredReferenceImplementation(int id) {
        this.id = id;
    }

    @Override
    public InstrumentedType prepare(InstrumentedType instrumentedType) {
        return instrumentedType;
    }

    @Override
    public ByteCodeAppender appender(Target implementationTarget) {
        return new RelativeLoadDeclaredInternalReference(id);
    }
}
