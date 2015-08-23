package org.nd4j.bytebuddy.method.integer.relative;

import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.member.MethodVariableAccess;
import net.bytebuddy.jar.asm.MethodVisitor;
import net.bytebuddy.jar.asm.Opcodes;

/**
 * @author Adam Gibson
 */
public class RelativeLoadIntParam implements ByteCodeAppender {
    private int offset = -1;

    public RelativeLoadIntParam(int offset) {
        this.offset = offset;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext, MethodDescription instrumentedMethod) {
        methodVisitor.visitIntInsn(Opcodes.ILOAD,offset);
        return new Size(1,1);
    }
}
