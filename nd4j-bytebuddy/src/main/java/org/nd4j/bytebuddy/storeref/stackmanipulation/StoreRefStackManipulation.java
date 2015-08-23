package org.nd4j.bytebuddy.storeref.stackmanipulation;

import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.jar.asm.MethodVisitor;
import net.bytebuddy.jar.asm.Opcodes;

/**
 * Stores a reference
 * in the specified id
 *
 * @author Adam Gibson
 */
public class StoreRefStackManipulation implements StackManipulation {
    private int storeId = -1;

    /**
     * The id to store in
     * @param storeId
     */
    public StoreRefStackManipulation(int storeId) {
        this.storeId = storeId;
    }

    @Override
    public boolean isValid() {
        return true;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext) {
        methodVisitor.visitVarInsn(Opcodes.ASTORE,storeId);
        return new Size(1, 1);
    }
}
