package org.nd4j.bytebuddy.arithmetic.relative;

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
public class RelativeIntegerArithmeticByteCodeAppender implements ByteCodeAppender {
    private int val1, val2;
    private StackManipulation op;

    /**
     *
     * @param val1 the first value to do the op on
     * @param val2 the second value to to do the op on
     * @param op the operation to perform
     */
    public RelativeIntegerArithmeticByteCodeAppender(int val1, int val2, RelativeByteBuddyIntArithmetic.Operation op) {
        this.val1 = val1;
        this.val2 = val2;
        this.op = RelativeByteBuddyIntArithmetic.opFor(op);
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext, MethodDescription instrumentedMethod) {
        if (!instrumentedMethod.getReturnType().represents(int.class)) {
            throw new IllegalArgumentException(instrumentedMethod + " must return int");
        }

        StackManipulation.Size operandStackSize = new StackManipulation.Compound(
                IntegerConstant.forValue(val1),
                IntegerConstant.forValue(val2),
                op,
                MethodReturn.INTEGER
        ).apply(methodVisitor, implementationContext);
        return new Size(operandStackSize.getMaximalSize(), instrumentedMethod.getStackSize());
    }
}
