package org.nd4j.linalg.ops;

import org.junit.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.reflections.Reflections;
import org.reflections.scanners.SubTypesScanner;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;
import org.reflections.util.FilterBuilder;

import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;
import java.util.Set;

import static org.junit.Assert.assertEquals;

public class OpConstructorTests extends BaseNd4jTest {

    public OpConstructorTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void checkForINDArrayConstructors() throws Exception {
        /*
        Check that all op classes have at least one INDArray or INDArray[] constructor, so they can actually
        be used outside of SameDiff
         */

        Reflections f = new Reflections(new ConfigurationBuilder()
                .filterInputsBy(new FilterBuilder().include(FilterBuilder.prefix("org.nd4j.*")).exclude("^(?!.*\\.class$).*$"))
                .setUrls(ClasspathHelper.forPackage("org.nd4j")).setScanners(new SubTypesScanner()));

        Set<Class<? extends DifferentialFunction>> classSet = f.getSubTypesOf(DifferentialFunction.class);

        int count = 0;
        for(Class<?> c : classSet){
            if(Modifier.isAbstract(c.getModifiers()) || Modifier.isInterface(c.getModifiers()) || c == SDVariable.class || ILossFunction.class.isAssignableFrom(c))
                continue;

//            System.out.println(c.getName());

            Constructor<?>[] constructors = c.getConstructors();
            boolean foundINDArray = false;
            for( int i=0; i<constructors.length; i++ ){
                Constructor<?> co = constructors[i];
                String str = co.toGenericString();      //This is a convenience hack for checking - returns strings like "public org.nd4j.linalg.api.ops.impl.reduce.floating.Norm2(org.nd4j.linalg.api.ndarray.INDArray,int...)"
                if(str.contains("INDArray") && !str.contains("SameDiff")){
                    foundINDArray = true;
                    break;
                }
            }

            if(!foundINDArray){
                System.out.println("No INDArray constructor: " + c.getName());
                count++;
            }
        }

        assertEquals(0, count);

    }

    @Override
    public char ordering(){
        return 'c';
    }

}
