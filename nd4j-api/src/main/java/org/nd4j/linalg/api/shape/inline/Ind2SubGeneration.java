package org.nd4j.linalg.api.shape.inline;

import net.bytebuddy.ByteBuddy;
import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.dynamic.DynamicType;
import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.constant.IntegerConstant;
import net.bytebuddy.implementation.bytecode.member.MethodReturn;
import net.bytebuddy.jar.asm.MethodVisitor;
import net.bytebuddy.jar.asm.Opcodes;
import org.nd4j.linalg.api.shape.Ind2Sub;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Adam Gibson
 */
public class Ind2SubGeneration {
    public enum IntegerMultiplication implements StackManipulation {

        INSTANCE;

        @Override
        public boolean isValid() {
            return true;
        }

        @Override
        public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext) {
            methodVisitor.visitInsn(Opcodes.IMUL);
            return new StackManipulation.Size(-1, 0);
        }
    }

    public enum IntegerDivision implements StackManipulation {

        INSTANCE;

        @Override
        public boolean isValid() {
            return true;
        }

        @Override
        public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext) {
            methodVisitor.visitInsn(Opcodes.IDIV);
            return new StackManipulation.Size(-1, 0);
        }
    }

    public enum IntegerMod implements StackManipulation {

        INSTANCE;

        @Override
        public boolean isValid() {
            return true;
        }

        @Override
        public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext) {
            methodVisitor.visitInsn(Opcodes.IREM);
            return new StackManipulation.Size(-1, 0);
        }
    }


    public static class Ind2SubImplementation implements Implementation {
        private int[] shape;
        private char order;

        public Ind2SubImplementation(int[] shape,char order) {
            this.shape = shape;
            this.order = order;
        }

        @Override
        public InstrumentedType prepare(InstrumentedType instrumentedType) {
            return instrumentedType;
        }

        private StackManipulation[] times() {
            List<StackManipulation[]> manipulations = new ArrayList<>();
            int denom = ArrayUtil.prod(shape);
            for(int i = 0; i < shape.length; i++) {
                manipulations.add(times(1,1));
            }

            return manipulations.toArray(new StackManipulation[manipulations.size()]);
        }

        private StackManipulation[] times(int val1,int val2) {
            return new StackManipulation[] {IntegerConstant.forValue(val1),
                    IntegerConstant.forValue(val2),
                    IntegerMultiplication.INSTANCE,
                    MethodReturn.INTEGER};
        }

        @Override
        public ByteCodeAppender appender(Target target) {
            return new ByteCodeAppender() {
                @Override
                public Size apply(MethodVisitor methodVisitor, Context implementationContext, MethodDescription instrumentedMethod) {
                    StackManipulation.Size operandStackSize = new StackManipulation.Compound(
                            IntegerConstant.forValue(10),
                            IntegerConstant.forValue(50),
                            IntegerMultiplication.INSTANCE,
                            MethodReturn.INTEGER
                    ).apply(methodVisitor, implementationContext);
                    return new Size(operandStackSize.getMaximalSize(), instrumentedMethod.getStackSize());
                }

            };
        }
    }

    public void generate(final char order,final int[] shape) {
        DynamicType.Unloaded<?> dynamicType = new ByteBuddy()
                .subclass(Ind2Sub.class).defineMethod("map", int[].class, Arrays.asList(int.class)).intercept(new Implementation() {
                    @Override
                    public InstrumentedType prepare(InstrumentedType instrumentedType) {
                        return instrumentedType;
                    }

                    @Override
                    public ByteCodeAppender appender(Target implementationTarget) {
                        return new Ind2SubImplementation(shape,order).appender(implementationTarget);
                    }
                }).make();
    }



    public static void main(String[] args) {
        Ind2SubGeneration generation = new Ind2SubGeneration();
        generation.generate('c',new int[]{2,2});
    }


}
