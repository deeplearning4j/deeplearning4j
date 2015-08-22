package org.nd4j.bytebuddy.arrays.create.noreturn;

import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.member.MethodReturn;
import net.bytebuddy.jar.asm.MethodVisitor;

/**
 * Byte code appender for creating
 * arrays of
 * the specified length
 * @author Adam Gibson
 */
public class CreateArrayByteCodeAppender implements ByteCodeAppender {
    private int length = -1;

    public CreateArrayByteCodeAppender(int length) {
        if(length < 0)
            throw new IllegalArgumentException("Unable to create array of length " + length);
        this.length = length;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext, MethodDescription instrumentedMethod) {
        StackManipulation createArray = IntArrayCreation.intCreationOfLength(length);
        StackManipulation.Compound size = new StackManipulation.Compound(
                createArray
        );

        StackManipulation.Size size1 = size.apply(methodVisitor, implementationContext);
        return new Size(size1.getMaximalSize(), instrumentedMethod.getStackSize());

    }
}
