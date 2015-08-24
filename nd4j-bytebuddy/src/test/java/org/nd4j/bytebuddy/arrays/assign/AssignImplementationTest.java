package org.nd4j.bytebuddy.arrays.assign;

import net.bytebuddy.ByteBuddy;
import net.bytebuddy.dynamic.DynamicType;
import net.bytebuddy.dynamic.loading.ClassLoadingStrategy;
import net.bytebuddy.implementation.bytecode.Duplication;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.member.MethodReturn;
import net.bytebuddy.implementation.bytecode.member.MethodVariableAccess;
import net.bytebuddy.matcher.ElementMatchers;
import org.junit.Test;
import org.nd4j.bytebuddy.arithmetic.stackmanipulation.OpStackManipulation;
import org.nd4j.bytebuddy.arrays.stackmanipulation.ArrayStackManipulation;
import org.nd4j.bytebuddy.stackmanipulation.StackManipulationImplementation;

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
                .intercept(new AssignImplmentation(0,1)).make().saveIn(new File("target/generated-classes"));
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

    @Test
    public void inPlaceSet() throws Exception {
        DynamicType.Unloaded<SetValueInPlace> val =  new ByteBuddy()
                .subclass(SetValueInPlace.class)
                .method(ElementMatchers.isDeclaredBy(SetValueInPlace.class))
                .intercept(new StackManipulationImplementation(
                        new StackManipulation.Compound(
                                MethodVariableAccess.REFERENCE.loadOffset(1),
                                MethodVariableAccess.INTEGER.loadOffset(2),
                                MethodVariableAccess.INTEGER.loadOffset(3),
                                ArrayStackManipulation.store(),
                                MethodReturn.VOID
                        ))).make();

        val.saveIn(new File("target"));
        SetValueInPlace dv =  val.load(getClass().getClassLoader(), ClassLoadingStrategy.Default.WRAPPER).getLoaded().newInstance();
        int[] ret = {2,4};
        int[] assertion = {1,4};
        dv.update(ret, 0, 1);
        assertArrayEquals(assertion, ret);
    }


    @Test
    public void inPlaceDivide() throws Exception {
        DynamicType.Unloaded<SetValueInPlace> val =  new ByteBuddy()
                .subclass(SetValueInPlace.class)
                .method(ElementMatchers.isDeclaredBy(SetValueInPlace.class))
                .intercept(new StackManipulationImplementation(
                        new StackManipulation.Compound(
                                MethodVariableAccess.REFERENCE.loadOffset(1),
                                MethodVariableAccess.INTEGER.loadOffset(2),
                                Duplication.DOUBLE,
                                ArrayStackManipulation.load(),
                                MethodVariableAccess.INTEGER.loadOffset(3),
                                OpStackManipulation.div(),
                                ArrayStackManipulation.store(),
                                MethodReturn.VOID
                        ))).make();

        val.saveIn(new File("target"));
        SetValueInPlace dv =  val.load(getClass().getClassLoader(), ClassLoadingStrategy.Default.WRAPPER).getLoaded().newInstance();
        int[] ret = {2,4};
        int[] assertion = {1,4};
        dv.update(ret,0,2);
        assertArrayEquals(assertion,ret);
    }

    public interface SetValueInPlace {
        void update(int[] values,int index,int divideBy);
    }

}
