package org.nd4j.bytebuddy.storeref;

import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.jar.asm.MethodVisitor;
import net.bytebuddy.jar.asm.Opcodes;

/**
 * @author Adam Gibson
 */
public class StoreRef implements ByteCodeAppender {
    private int storeId = -1;

    public StoreRef(int storeId) {
        this.storeId = storeId;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext, MethodDescription instrumentedMethod) {
        methodVisitor.visitInsn(Opcodes.AASTORE);
        return new Size(instrumentedMethod.getStackSize(), 1);

    }
}
