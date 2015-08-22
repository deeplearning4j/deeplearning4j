package org.nd4j.bytebuddy.aggregate;

import net.bytebuddy.ByteBuddy;
import net.bytebuddy.dynamic.DynamicType;
import net.bytebuddy.dynamic.loading.ClassLoadingStrategy;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.matcher.ElementMatchers;
import org.junit.Test;
import org.nd4j.bytebuddy.arrays.assign.relative.RelativeAssignImplementation;
import org.nd4j.bytebuddy.arrays.create.noreturn.IntArrayCreation;
import org.nd4j.bytebuddy.dup.DuplicateImplementation;
import org.nd4j.bytebuddy.returnref.ReturnAppender;
import org.nd4j.bytebuddy.returnref.ReturnAppenderImplementation;


import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class TestAggregateByteCodeAppender {
    @Test
    public void testCreateAndAssign() throws Exception {
        DynamicType.Unloaded<CreateAndAssignArray> arr = new ByteBuddy()
                .subclass(CreateAndAssignArray.class).method(ElementMatchers.isDeclaredBy(CreateAndAssignArray.class))
                .intercept(new Implementation.Compound(
                        new IntArrayCreation(5),
                        new DuplicateImplementation(),
                        new RelativeAssignImplementation(0, 5),
                        new ReturnAppenderImplementation(ReturnAppender.ReturnType.REFERENCE)))
                .make();


        Class<?> dynamicType = arr.load(CreateAndAssignArray.class.getClassLoader(), ClassLoadingStrategy.Default.WRAPPER)
                .getLoaded();

        CreateAndAssignArray test = (CreateAndAssignArray) dynamicType.newInstance();
        int[] result = test.create();
        assertEquals(5,result[0]);
    }

    public interface CreateAndAssignArray {
        int[] create();
    }
}
