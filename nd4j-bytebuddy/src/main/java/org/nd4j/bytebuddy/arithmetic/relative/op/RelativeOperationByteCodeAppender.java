package org.nd4j.bytebuddy.arithmetic.relative.op;

import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.constant.IntegerConstant;
import net.bytebuddy.implementation.bytecode.member.MethodReturn;
import net.bytebuddy.jar.asm.MethodVisitor;

/**
 * Appends byte code for basic operations between 2 integers:
 * add
 * sub
 * mul
 * div
 * mod
 *
 * @author Adam Gibson
 */
public class RelativeOperationByteCodeAppender implements ByteCodeAppender {
    private StackManipulation op;

    /**
     *
     * * @param op the operation to perform
     */
    public RelativeOperationByteCodeAppender(RelativeOperationImplementation.Operation op) {
        this.op = RelativeOperationImplementation.opFor(op);
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext, MethodDescription instrumentedMethod) {
        StackManipulation.Size size = op.apply(methodVisitor, implementationContext);
        return new Size(size.getMaximalSize(), instrumentedMethod.getStackSize());
    }
}
