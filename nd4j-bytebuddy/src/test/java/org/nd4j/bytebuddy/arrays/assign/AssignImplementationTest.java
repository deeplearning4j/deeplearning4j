package org.nd4j.bytebuddy.arrays.assign;

import net.bytebuddy.ByteBuddy;
import net.bytebuddy.dynamic.loading.ClassLoadingStrategy;
import net.bytebuddy.matcher.ElementMatchers;
import org.junit.Test;

import java.io.File;

import static org.junit.Assert.*;
/**
 * @author Adam Gibson
 */
public class AssignImplementationTest {

    @Test
    public void testAssign() throws Exception {
        new ByteBuddy()
                .subclass(AssignValue.class).method(ElementMatchers.isDeclaredBy(AssignValue.class))
                .intercept(new AssignImplmentation(0,1)).make().saveIn(new File("/home/agibsonccc/Desktop"));
        Class<?> dynamicType = new ByteBuddy()
                .subclass(AssignValue.class).method(ElementMatchers.isDeclaredBy(AssignValue.class))
                .intercept(new AssignImplmentation(0,1)).make()
                .load(AssignValue.class.getClassLoader(), ClassLoadingStrategy.Default.WRAPPER)
                .getLoaded();
        int[] vals = new int[2];
        AssignValue instance = (AssignValue) dynamicType.newInstance();
        instance.assign(vals,0,1);
        assertEquals(1,vals[0]);
    }

}
