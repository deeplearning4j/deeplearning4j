package org.nd4j.bytebuddy.arithmetic.relative.op;

import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.constant.IntegerConstant;
import net.bytebuddy.implementation.bytecode.member.MethodReturn;
import net.bytebuddy.jar.asm.MethodVisitor;
import net.bytebuddy.jar.asm.Opcodes;


/**
 * Handles actual arithmetic
 * between 2 numbers:
 * add
 * sub
 * mul
 * div
 * mod
 *
 * @author Adam Gibson
 */
public class RelativeOperationImplementation implements Implementation {
    private Operation op;


    /**
     * and an operation to do on them
     * @param op
     */
    public RelativeOperationImplementation(Operation op) {
        this.op = op;
    }


    /**
     * Returns the proper stack manipulation
     * for the given operation
     * @param operation the arithmetic operation to do
     * @return the stack manipulation for the given operation
     */
    public static StackManipulation opFor(Operation operation) {
        switch(operation) {
            case ADD: return IntegerAddition.INSTANCE;
            case SUB: return IntegerSubtraction.INSTANCE;
            case MUL: return IntegerMultiplication.INSTANCE;
            case DIV: return IntegerDivision.INSTANCE;
            case MOD: return IntegerMod.INSTANCE;
            default: throw new IllegalArgumentException("Illegal type of operation ");
        }
    }


    @Override
    public InstrumentedType prepare(InstrumentedType instrumentedType) {
        return instrumentedType;
    }

    @Override
    public ByteCodeAppender appender(Target implementationTarget) {
        return new RelativeOperationByteCodeAppender(op);
    }

    public enum Operation {
        ADD,SUB,MUL,DIV,MOD
    }

    public enum IntegerSubtraction implements StackManipulation {

        INSTANCE;

        @Override
        public boolean isValid() {
            return true;
        }

        @Override
        public Size apply(MethodVisitor methodVisitor, Context implementationContext) {
            methodVisitor.visitInsn(Opcodes.ISUB);
            return new Size(-1, 0);
        }
    }

    public enum IntegerAddition implements StackManipulation {

        INSTANCE;

        @Override
        public boolean isValid() {
            return true;
        }

        @Override
        public Size apply(MethodVisitor methodVisitor, Context implementationContext) {
            methodVisitor.visitInsn(Opcodes.IADD);
            return new Size(-1, 0);
        }
    }

    public enum IntegerMultiplication implements StackManipulation {

        INSTANCE;

        @Override
        public boolean isValid() {
            return true;
        }

        @Override
        public Size apply(MethodVisitor methodVisitor, Context implementationContext) {
            methodVisitor.visitInsn(Opcodes.IMUL);
            return new Size(-1, 0);
        }
    }

    public enum IntegerDivision implements StackManipulation {

        INSTANCE;

        @Override
        public boolean isValid() {
            return true;
        }

        @Override
        public Size apply(MethodVisitor methodVisitor, Context implementationContext) {
            methodVisitor.visitInsn(Opcodes.IDIV);
            return new Size(-1, 0);
        }
    }

    public enum IntegerMod implements StackManipulation {

        INSTANCE;

        @Override
        public boolean isValid() {
            return true;
        }

        @Override
        public Size apply(MethodVisitor methodVisitor, Context implementationContext) {
            methodVisitor.visitInsn(Opcodes.IREM);
            return new Size(-1, 0);
        }
    }




}
