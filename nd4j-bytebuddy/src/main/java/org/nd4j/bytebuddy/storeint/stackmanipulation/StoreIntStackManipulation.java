package org.nd4j.bytebuddy.storeint.stackmanipulation;

import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.jar.asm.MethodVisitor;
import net.bytebuddy.jar.asm.Opcodes;

/**
 * @author Adam Gibson
 */
public class StoreIntStackManipulation implements StackManipulation {
    private int storeId = -1;

    public StoreIntStackManipulation(int storeId) {
        this.storeId = storeId;
    }

    @Override
    public boolean isValid() {
        return true;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext) {
        methodVisitor.visitVarInsn(Opcodes.ISTORE,storeId);
        return new Size(1, 1);
    }
}
