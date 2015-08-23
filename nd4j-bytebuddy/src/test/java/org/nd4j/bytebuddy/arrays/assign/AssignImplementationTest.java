package org.nd4j.bytebuddy.arrays.assign;

import com.sun.org.apache.xpath.internal.operations.Div;
import net.bytebuddy.ByteBuddy;
import net.bytebuddy.dynamic.DynamicType;
import net.bytebuddy.dynamic.loading.ClassLoadingStrategy;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.member.MethodReturn;
import net.bytebuddy.implementation.bytecode.member.MethodVariableAccess;
import net.bytebuddy.matcher.ElementMatchers;
import org.junit.Test;
import org.nd4j.bytebuddy.arithmetic.relative.op.RelativeOperationImplementation;
import org.nd4j.bytebuddy.arrays.assign.relative.novalue.RelativeAssignNoValueImplementation;
import org.nd4j.bytebuddy.arrays.assign.relative.novalue.noindex.ArrayStoreImplementation;
import org.nd4j.bytebuddy.arrays.retrieve.relative.RelativeRetrieveArrayImplementation;
import org.nd4j.bytebuddy.arrays.stackmanipulation.ArrayStackManipulation;
import org.nd4j.bytebuddy.constant.ConstantIntImplementation;
import org.nd4j.bytebuddy.dup.Duplicate2Implementation;
import org.nd4j.bytebuddy.dup.DuplicateImplementation;
import org.nd4j.bytebuddy.loadref.relative.RelativeLoadDeclaredReferenceImplementation;
import org.nd4j.bytebuddy.method.args.LoadArgsImplementation;
import org.nd4j.bytebuddy.method.integer.LoadIntParamImplementation;
import org.nd4j.bytebuddy.method.reference.LoadReferenceParamImplementation;
import org.nd4j.bytebuddy.returnref.ReturnAppender;
import org.nd4j.bytebuddy.returnref.ReturnAppenderImplementation;
import org.nd4j.bytebuddy.stackmanipulation.StackManipulationImplementation;

import java.io.File;
import java.lang.reflect.Method;

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
    public void inPlaceDivision() throws Exception {
        DynamicType.Unloaded<DivideInPlace> val =  new ByteBuddy()
                .subclass(DivideInPlace.class)
                .method(ElementMatchers.isDeclaredBy(DivideInPlace.class))
                .intercept(new StackManipulationImplementation(
                        new StackManipulation.Compound(
                                MethodVariableAccess.REFERENCE.loadOffset(1),
                                MethodVariableAccess.INTEGER.loadOffset(2),
                                MethodVariableAccess.INTEGER.loadOffset(3),
                                ArrayStackManipulation.store(),
                                MethodReturn.VOID
                        ))).make();

        val.saveIn(new File("target"));
        DivideInPlace dv =  val.load(getClass().getClassLoader(), ClassLoadingStrategy.Default.WRAPPER).getLoaded().newInstance();
        int[] ret = {2,4};
        int[] assertion = {1,4};
        dv.update(ret,0,1);
        assertArrayEquals(assertion,ret);
    }

    public interface DivideInPlace {
        void update(int[] values,int index,int divideBy);
    }

}
