package org.nd4j.bytebuddy.arrays;

import net.bytebuddy.ByteBuddy;
import net.bytebuddy.dynamic.loading.ClassLoadingStrategy;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.matcher.ElementMatchers;
import org.junit.Test;
import org.nd4j.bytebuddy.arrays.create.CreateArrayByteCodeAppender;
import org.nd4j.bytebuddy.arrays.create.IntArrayCreation;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Adam Gibson
 */
public class IntArrayCreationTest {
    @Test
    public void testStackManipulationForLength() {
        assertTrue(IntArrayCreation.intCreationOfLength(5).isValid());
    }

    @Test
    public void testArrayCreationByteCode() {
        CreateArrayByteCodeAppender append = new CreateArrayByteCodeAppender(5);
        StackManipulation manipulation = append.stackManipulationForLength();
        assertTrue(manipulation.isValid());

    }

    @Test
    public void testCreateArrayOfLength2() throws Exception {
        Class<?> dynamicType = new ByteBuddy()
                .subclass(CreateArray.class).method(ElementMatchers.isDeclaredBy(CreateArray.class))
                .intercept(new IntArrayCreation(5)).make()
                .load(CreateArray.class.getClassLoader(), ClassLoadingStrategy.Default.WRAPPER)
                .getLoaded();
        CreateArray addition = (CreateArray) dynamicType.newInstance();
        int[] arr2 = addition.create();
        assertEquals(5,arr2.length);
    }

    public interface CreateArray {
        int[] create();
    }


}
