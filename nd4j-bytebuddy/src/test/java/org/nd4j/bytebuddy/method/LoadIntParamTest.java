package org.nd4j.bytebuddy.method;

import net.bytebuddy.ByteBuddy;
import net.bytebuddy.dynamic.loading.ClassLoadingStrategy;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.matcher.ElementMatchers;
import org.junit.Test;
import org.nd4j.bytebuddy.method.integer.LoadIntParamImplementation;
import org.nd4j.bytebuddy.returnref.ReturnAppender;
import org.nd4j.bytebuddy.returnref.ReturnAppenderImplementation;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class LoadIntParamTest {
    @Test
    public void testLoadParam() throws Exception {
        //note the 2 here: the indexing is as follows
        //this arg0, arg1
        //the method indexing always starts at zero with this
        Class<?> dynamicType = new ByteBuddy()
                .subclass(GrabArgOne.class).method(ElementMatchers.isDeclaredBy(GrabArgOne.class))
                .intercept(new Implementation.Compound(
                        new LoadIntParamImplementation(2)
                        ,new ReturnAppenderImplementation(ReturnAppender.ReturnType.INT))).make()
                .load(GrabArgOne.class.getClassLoader(), ClassLoadingStrategy.Default.WRAPPER)
                .getLoaded();
        GrabArgOne argOne = (GrabArgOne) dynamicType.newInstance();
        int val = argOne.ret(1,2);
        assertEquals(2,val);

        Class<?> dynamicType2 = new ByteBuddy()
                .subclass(GrabArgOne.class).method(ElementMatchers.isDeclaredBy(GrabArgOne.class))
                .intercept(new Implementation.Compound(
                        new LoadIntParamImplementation(1)
                        ,new ReturnAppenderImplementation(ReturnAppender.ReturnType.INT))).make()
                .load(GrabArgOne.class.getClassLoader(), ClassLoadingStrategy.Default.WRAPPER)
                .getLoaded();
        GrabArgOne argOne2 = (GrabArgOne) dynamicType2.newInstance();
        int val2 = argOne2.ret(1,2);
        assertEquals(1,val2);


    }

    public interface GrabArgOne {
        int ret(int arg0,int arg1);
    }

}
