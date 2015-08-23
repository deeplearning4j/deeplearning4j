package org.nd4j.bytebuddy.method.args;

import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.member.MethodVariableAccess;
import net.bytebuddy.jar.asm.MethodVisitor;

/**
 * @author Adam Gibson
 */
public class LoadArgsAppender implements ByteCodeAppender {
    private boolean loadThis = false;

    public LoadArgsAppender(boolean loadThis) {
        this.loadThis = loadThis;
    }

    public LoadArgsAppender() {
        this(false);
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext, MethodDescription instrumentedMethod) {
        StackManipulation.Size size = loadThis ? MethodVariableAccess.loadThisReferenceAndArguments(instrumentedMethod).apply(methodVisitor, implementationContext) : MethodVariableAccess.loadArguments(instrumentedMethod).apply(methodVisitor, implementationContext);
        return new Size(size.getMaximalSize(), instrumentedMethod.getStackSize());
    }
}
