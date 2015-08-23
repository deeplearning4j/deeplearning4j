package org.nd4j.bytebuddy.arrays.create.stackmanipulation;

import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.jar.asm.MethodVisitor;
import net.bytebuddy.jar.asm.Opcodes;

/**
 * @author Adam Gibson
 */
public class CreateIntArrayStackManipulation implements StackManipulation {
    @Override
    public boolean isValid() {
        return true;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext) {
        methodVisitor.visitIntInsn(Opcodes.NEWARRAY,Opcodes.T_INT);
        return new Size(1,1);
    }
}
