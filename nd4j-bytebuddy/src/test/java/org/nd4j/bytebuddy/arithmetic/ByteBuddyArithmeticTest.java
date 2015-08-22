package org.nd4j.bytebuddy.arithmetic;

import net.bytebuddy.ByteBuddy;
import net.bytebuddy.dynamic.loading.ClassLoadingStrategy;
import net.bytebuddy.matcher.ElementMatchers;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class ByteBuddyArithmeticTest {
    @Test
    public void testOperations() throws Exception {
        int[] results = new int[]{
                5,1,6,1,1
        };

        //ADD,SUB,MUL,DIV,MOD
        ByteBuddyIntArithmetic.Operation[] ops = ByteBuddyIntArithmetic.Operation.values();


        for(int i = 0; i < results.length; i++) {
            Class<?> dynamicType = new ByteBuddy()
                    .subclass(Arithmetic.class).method(ElementMatchers.isDeclaredBy(Arithmetic.class))
                    .intercept(new ByteBuddyIntArithmetic(3, 2, ops[i])).make()
                    .load(Arithmetic.class.getClassLoader(), ClassLoadingStrategy.Default.WRAPPER)
                    .getLoaded();
            Arithmetic addition = (Arithmetic) dynamicType.newInstance();
            assertEquals("Failed on " + i, results[i], addition.calc());
        }

    }



    public interface Arithmetic {
        int calc();
    }

}
