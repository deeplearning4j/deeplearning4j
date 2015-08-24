package org.nd4j.bytebuddy.storeint;

import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.member.MethodVariableAccess;
import net.bytebuddy.jar.asm.MethodVisitor;
import net.bytebuddy.jar.asm.Opcodes;

/**
 * @author Adam Gibson
 */
public class StoreInt implements ByteCodeAppender {
    private int storeId = -1;

    public StoreInt(int storeId) {
        this.storeId = storeId;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext, MethodDescription instrumentedMethod) {
        methodVisitor.visitVarInsn(Opcodes.ISTORE,storeId);
        return new Size(instrumentedMethod.getStackSize(), 1);

    }
}
