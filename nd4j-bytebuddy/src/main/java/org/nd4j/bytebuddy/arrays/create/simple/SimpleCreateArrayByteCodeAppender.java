package org.nd4j.bytebuddy.arrays.create.simple;

import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.constant.IntegerConstant;
import net.bytebuddy.jar.asm.MethodVisitor;
import net.bytebuddy.jar.asm.Opcodes;

/**
 * @author Adam Gibson
 */
public class SimpleCreateArrayByteCodeAppender implements ByteCodeAppender {
    private int length = -1;

    public SimpleCreateArrayByteCodeAppender(int length) {
        this.length = length;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext, MethodDescription instrumentedMethod) {
        StackManipulation.Size size = IntegerConstant.forValue(length).apply(methodVisitor, implementationContext);
        methodVisitor.visitIntInsn(Opcodes.NEWARRAY,Opcodes.T_INT);
        return new Size(size.getMaximalSize(),size.getSizeImpact());
    }
}
