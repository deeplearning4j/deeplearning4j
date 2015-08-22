package org.nd4j.bytebuddy.dup;

import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.jar.asm.MethodVisitor;
import net.bytebuddy.jar.asm.Opcodes;

/**
 * @author Adam Gibson
 */
public class Duplicate implements ByteCodeAppender {
    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext, MethodDescription instrumentedMethod) {
        methodVisitor.visitInsn(Opcodes.DUP);
        return new Size(1,1);
    }
}
