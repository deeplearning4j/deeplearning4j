package org.nd4j.bytebuddy.method.integer;

import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.member.MethodVariableAccess;
import net.bytebuddy.jar.asm.MethodVisitor;

/**
 * @author Adam Gibson
 */
public class LoadIntParam implements ByteCodeAppender {
    private int offset = -1;

    public LoadIntParam(int offset) {
        this.offset = offset;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext, MethodDescription instrumentedMethod) {
        StackManipulation.Size size = MethodVariableAccess.INTEGER.loadOffset(offset).apply(methodVisitor, implementationContext);
        return new Size(size.getMaximalSize(),instrumentedMethod.getStackSize());
    }
}
