package org.nd4j.bytebuddy.dup;

import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.Duplication;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.jar.asm.MethodVisitor;
import net.bytebuddy.jar.asm.Opcodes;

/**
 * http://cs.au.dk/~mis/dOvs/jvmspec/ref-_dup2.html
 * @author Adam Gibson
 */
public class Duplicate2 implements ByteCodeAppender {
    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext, MethodDescription instrumentedMethod) {
        StackManipulation.Size size = Duplication.DOUBLE.apply(methodVisitor, implementationContext);
        return new Size(size.getMaximalSize(),instrumentedMethod.getStackSize());
    }
}
