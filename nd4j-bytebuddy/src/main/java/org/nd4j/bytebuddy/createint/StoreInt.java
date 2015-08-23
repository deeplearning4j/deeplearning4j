package org.nd4j.bytebuddy.createint;

import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.jar.asm.MethodVisitor;
import net.bytebuddy.jar.asm.Opcodes;

/**
 * Stores an int in a variable
 *
 * @author Adam Gibson
 */
public class StoreInt implements ByteCodeAppender {
    private int idx = -1;

    public StoreInt(int idx) {
        this.idx = idx;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext, MethodDescription instrumentedMethod) {
        if (!instrumentedMethod.getReturnType().represents(int.class)) {
            throw new IllegalArgumentException(instrumentedMethod + " must return int");
        }

        methodVisitor.visitVarInsn(Opcodes.ISTORE, idx);
        //add 1 for the store operation
        return new Size(1, 1);
    }


}
