package org.nd4j.bytebuddy.returnref;

import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.member.MethodReturn;
import net.bytebuddy.jar.asm.MethodVisitor;

/**
 * Appends a return statement to
 * a stack frame
 *
 * @author Adam Gibson
 */
public class ReturnAppender implements ByteCodeAppender {
    public enum ReturnType {
        REFERENCE
        ,VOID,
        INT
    }

    private ReturnType returnType;

    public ReturnAppender(ReturnType returnType) {
        this.returnType = returnType;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext, MethodDescription instrumentedMethod) {
        switch (returnType) {
            case REFERENCE: return new Size(MethodReturn.REFERENCE.apply(methodVisitor,implementationContext).getMaximalSize(),instrumentedMethod.getStackSize());
            case VOID:  return new Size(MethodReturn.VOID.apply(methodVisitor,implementationContext).getMaximalSize(),instrumentedMethod.getStackSize());
            case INT:  return new Size(MethodReturn.INTEGER.apply(methodVisitor,implementationContext).getMaximalSize(),instrumentedMethod.getStackSize());
            default: throw new IllegalStateException("Illegal type");
        }

    }
}
