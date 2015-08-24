package org.nd4j.bytebuddy.stackmanipulation;

import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.jar.asm.MethodVisitor;

/**
 * @author Adam Gibson
 */
public class StackManipulationByteCodeAppender implements ByteCodeAppender {
    private StackManipulation stackManipulation;

    public StackManipulationByteCodeAppender(StackManipulation stackManipulation) {
        this.stackManipulation = stackManipulation;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext, MethodDescription instrumentedMethod) {
        StackManipulation.Size size =  stackManipulation.apply(methodVisitor, implementationContext);
        return new Size(size.getMaximalSize(),instrumentedMethod.getStackSize());
    }
}
