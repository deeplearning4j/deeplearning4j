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
    public void testAddition() throws Exception {
        Class<?> dynamicType = new ByteBuddy()
                .subclass(Addition.class).method(ElementMatchers.isDeclaredBy(Addition.class))
                .intercept(new ByteBuddyIntArithmetic(3, 2, ByteBuddyIntArithmetic.Operation.ADD)).make()
                .load(Addition.class.getClassLoader(), ClassLoadingStrategy.Default.INJECTION)
                .getLoaded();
        Addition addition = (Addition) dynamicType.newInstance();
        assertEquals(5,addition.add());
    }


    public interface Addition {
        int add();
    }

}
