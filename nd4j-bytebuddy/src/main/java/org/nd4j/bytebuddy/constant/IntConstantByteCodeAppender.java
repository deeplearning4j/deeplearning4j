package org.nd4j.bytebuddy.constant;

import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.constant.IntegerConstant;
import net.bytebuddy.jar.asm.MethodVisitor;

/**
 *
 * @author Adam Gibson
 */
public class IntConstantByteCodeAppender implements ByteCodeAppender {
    private int constantVal = -1;

    public IntConstantByteCodeAppender(int constantVal) {
        this.constantVal = constantVal;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext, MethodDescription instrumentedMethod) {
        StackManipulation.Size size = IntegerConstant.forValue(constantVal).apply(methodVisitor, implementationContext);
        return new Size(size.getMaximalSize(),instrumentedMethod.getStackSize());
    }
}
