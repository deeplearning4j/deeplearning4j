package org.nd4j.bytebuddy.dup;

import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.Duplication;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.jar.asm.MethodVisitor;

/**
 * @author Adam Gibson
 */
public class Duplicate implements ByteCodeAppender {
    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext,
                    MethodDescription instrumentedMethod) {
        StackManipulation.Size size = Duplication.SINGLE.apply(methodVisitor, implementationContext);
        return new Size(size.getMaximalSize(), instrumentedMethod.getStackSize());
    }
}
