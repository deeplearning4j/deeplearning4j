package org.nd4j.bytebuddy.arrays.create;

import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.collection.ArrayFactory;
import net.bytebuddy.implementation.bytecode.constant.IntegerConstant;
import net.bytebuddy.pool.TypePool;

import java.util.ArrayList;
import java.util.List;

/**
 * Creates int arrays
 *
 * @author Adam Gibson
 */
public class IntArrayCreation implements Implementation {
    private int length = 0;
    private static TypePool typePool = TypePool.Default.ofClassPath();
    private static ArrayFactory factory = ArrayFactory.forType(typePool.describe("int").resolve());

    public IntArrayCreation(int length) {
        this.length = length;
    }

    @Override
    public InstrumentedType prepare(InstrumentedType instrumentedType) {
        return instrumentedType;
    }

    @Override
    public ByteCodeAppender appender(Target implementationTarget) {
        return new CreateArrayByteCodeAppender(length);
    }


    public static StackManipulation intCreationOfLength(int length) {
        List<StackManipulation> manipulations = new ArrayList<>();
        for(int i = 0; i < length; i++)
            manipulations.add(IntegerConstant.forValue(0));
        StackManipulation createArray = factory.withValues(manipulations);
        return createArray;

    }

}
