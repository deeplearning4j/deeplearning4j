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
 * of a specified length
 *
 *
 * @author Adam Gibson
 */
public class IntArrayCreation implements Implementation {
    private int length = 0;
    private static TypePool typePool = TypePool.Default.ofClassPath();
    private static ArrayFactory factory = ArrayFactory.forType(typePool.describe("int").resolve());

    /**
     * Specify the length
     * of the array to create
     * @param length the length of the array to create
     */
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


    /**
     * Creates te stack manipulation for an array
     * of the given length
     * @param length the length of the array
     * @return the stack manipulation representing
     * the array creation of the specified length
     */
    public static StackManipulation intCreationOfLength(int length) {
        List<StackManipulation> manipulations = new ArrayList<>();
        for(int i = 0; i < length; i++)
            manipulations.add(IntegerConstant.forValue(0));
        StackManipulation createArray = factory.withValues(manipulations);
        return createArray;

    }

}
