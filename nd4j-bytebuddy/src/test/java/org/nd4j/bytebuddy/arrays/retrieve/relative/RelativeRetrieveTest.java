package org.nd4j.bytebuddy.arrays.retrieve.relative;

import net.bytebuddy.ByteBuddy;
import net.bytebuddy.dynamic.DynamicType;
import net.bytebuddy.dynamic.loading.ClassLoadingStrategy;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.matcher.ElementMatchers;
import org.junit.Test;
import org.nd4j.bytebuddy.method.reference.LoadReferenceParamImplementation;
import org.nd4j.bytebuddy.returnref.ReturnAppender;
import org.nd4j.bytebuddy.returnref.ReturnAppenderImplementation;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class RelativeRetrieveTest {
    @Test
    public void testRetrieveFromArray() throws Exception {
        DynamicType.Unloaded<RetrieveFromArray> arr = new ByteBuddy()
                .subclass(RetrieveFromArray.class).method(ElementMatchers.isDeclaredBy(RetrieveFromArray.class))
                .intercept(new Implementation.Compound(
                        new LoadReferenceParamImplementation(1),
                        new RelativeRetrieveArrayImplementation(1),
                        new ReturnAppenderImplementation(ReturnAppender.ReturnType.INT)))
                .make();


        Class<?> dynamicType = arr.load(RetrieveFromArray.class.getClassLoader(), ClassLoadingStrategy.Default.WRAPPER)
                .getLoaded();

        RetrieveFromArray test = (RetrieveFromArray) dynamicType.newInstance();
        int result = test.returnVal(0,1);
        assertEquals(1,result);

    }

    public interface RetrieveFromArray {
        int returnVal(int...array);
    }


}
