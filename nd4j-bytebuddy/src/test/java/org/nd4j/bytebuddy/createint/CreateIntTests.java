package org.nd4j.bytebuddy.createint;

import net.bytebuddy.ByteBuddy;
import net.bytebuddy.dynamic.DynamicType;
import net.bytebuddy.dynamic.loading.ClassLoadingStrategy;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.matcher.ElementMatchers;
import org.junit.Test;
import org.nd4j.bytebuddy.constant.ConstantIntImplementation;
import org.nd4j.bytebuddy.load.LoadIntegerImplementation;
import org.nd4j.bytebuddy.returnref.ReturnAppender;
import org.nd4j.bytebuddy.returnref.ReturnAppenderImplementation;

import java.io.File;

import static org.junit.Assert.assertEquals;

/**
 * @author Adam Gibson
 */
public class CreateIntTests {

    @Test
    public void testCreateInt() throws Exception {
        DynamicType.Unloaded<CreateAndAssignIntArray> arr = new ByteBuddy()
                .subclass(CreateAndAssignIntArray.class).method(ElementMatchers.isDeclaredBy(CreateAndAssignIntArray.class))
                .intercept(new Implementation.Compound(
                        new ConstantIntImplementation(1),
                        new StoreIntImplementation(0),
                        new LoadIntegerImplementation(0),
                        new ReturnAppenderImplementation(ReturnAppender.ReturnType.INT)))
                .make();


        Class<?> dynamicType = arr.load(CreateAndAssignIntArray.class.getClassLoader(), ClassLoadingStrategy.Default.WRAPPER)
                .getLoaded();

        CreateAndAssignIntArray test = (CreateAndAssignIntArray) dynamicType.newInstance();
        int result = test.returnVal();
        assertEquals(1,result);

    }

    public interface CreateAndAssignIntArray {
        int returnVal();
    }

}
