package org.nd4j.bytebuddy.arithmetic;

import net.bytebuddy.ByteBuddy;
import net.bytebuddy.dynamic.DynamicType;
import net.bytebuddy.dynamic.loading.ClassLoadingStrategy;
import net.bytebuddy.dynamic.scaffold.InstrumentedType;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.matcher.ElementMatchers;
import org.junit.Test;

import java.util.Arrays;

/**
 * @author Adam Gibson
 */
public class ByteBuddyArithmeticTest {
    @Test
    public void testAddition() throws Exception {
        final IntegerArithmeticByteCodeAppender appender = new IntegerArithmeticByteCodeAppender(1,2, ByteBuddyIntArithmetic.Operation.ADD);
        Class<?> dynamicType = new ByteBuddy()
                .subclass(Addition.class).method(ElementMatchers.isDeclaredBy(Addition.class))
                .intercept(new ByteBuddyIntArithmetic(3, 2, ByteBuddyIntArithmetic.Operation.ADD)).make()
                .load(Addition.class.getClassLoader(), ClassLoadingStrategy.Default.INJECTION)
                .getLoaded();
        Addition addition = (Addition) dynamicType.newInstance();
        System.out.println(addition.add(1, 2));
    }


    public interface Addition {
        int add(int first,int second);
    }

}
