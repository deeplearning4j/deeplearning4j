package org.nd4j.bytebuddy.loadref;

import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.member.MethodVariableAccess;
import net.bytebuddy.jar.asm.MethodVisitor;
import org.nd4j.bytebuddy.util.OpCodeUtil;

/**
 * Load a reference on the method stack
 * relative to the number of arguments
 * in the method.
 *
 * This is meant for loading certain variable
 * declarations from the stack that were likely created
 * as part of some control flow or independent inline code.
 *
 * An example of this would be:
 * int[] create(int arg0) {
 *     int[] ret = new int[5];
 * }
 *
 * This method will not load arg0 but instead ret because it was the variable declared.
 *
 * @author Adam Gibson
 */
public class LoadDeclaredInternalReference implements ByteCodeAppender {
    private int refId = -1;

    /**
     * Pass in a ref id
     * for loading a reference off the stack.
     * @param refId
     */
    public LoadDeclaredInternalReference(int refId) {
        this.refId = refId;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext, MethodDescription instrumentedMethod) {
        int numArgs = instrumentedMethod.getParameters().asTypeList().getStackSize();
        /**
         * Load the desired id
         * relative to the method arguments.
         * The idea here would be to load references
         * to declared variables
         */
        //references start with zero if its an instance or zero if its static
        //think of it like an implicit self in python without actually being defined
        int start = instrumentedMethod.isStatic() ? 1 : 0;
        StackManipulation arg0 = MethodVariableAccess.REFERENCE.loadOffset(numArgs + start + refId);
        StackManipulation.Size size =  arg0.apply(methodVisitor, implementationContext);
        return new Size(size.getMaximalSize(), instrumentedMethod.getStackSize());
    }
}
