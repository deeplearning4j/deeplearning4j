package org.nd4j.autodiff;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.converters.ImportClassMapping;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.reflections.Reflections;

import java.lang.reflect.Modifier;
import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestOpMapping {

    @Test
    public void testOpMappingCoverage() throws Exception {
        Reflections reflections = new Reflections("org.nd4j");
        Set<Class<? extends DifferentialFunction>> subTypes = reflections.getSubTypesOf(DifferentialFunction.class);

        Map<String, DifferentialFunction> opNameMapping = ImportClassMapping.getOpNameMapping();
        Map<String, DifferentialFunction> tfOpNameMapping = ImportClassMapping.getTFOpMappingFunctions();
        Map<String, DifferentialFunction> onnxOpNameMapping = ImportClassMapping.getOnnxOpMappingFunctions();


        for(Class<? extends DifferentialFunction> c : subTypes){

            if(Modifier.isAbstract(c.getModifiers()) || Modifier.isInterface(c.getModifiers()) || c == SDVariable.class || ILossFunction.class.isAssignableFrom(c))
                continue;

            DifferentialFunction df = c.newInstance();
            String opName = df.opName();

            assertTrue(opName, opNameMapping.containsKey(opName));

            try{
                String[] tfNames = df.tensorflowNames();

                for(String s : tfNames ){
                    assertTrue("Tensorflow mapping not found: " + s, tfOpNameMapping.containsKey(s));
                    assertEquals("Tensorflow mapping: " + s, df.getClass(), tfOpNameMapping.get(s).getClass());
                }
            } catch (NoOpNameFoundException e){
                //OK, skip
            }


            try{
                String[] onnxNames = df.onnxNames();

                for(String s : onnxNames ){
                    assertTrue("Onnx mapping not found: " + s, onnxOpNameMapping.containsKey(s));
                    assertEquals("Onnx mapping: " + s, df.getClass(), onnxOpNameMapping.get(s).getClass());
                }
            } catch (NoOpNameFoundException e){
                //OK, skip
            }
        }
    }


    @Test @Ignore
    public void generateOpClassList() throws Exception{
        Reflections reflections = new Reflections("org.nd4j");
        Set<Class<? extends DifferentialFunction>> subTypes = reflections.getSubTypesOf(DifferentialFunction.class);

        List<Class<?>> l = new ArrayList<>();
        for(Class<?> c : subTypes){
            if(Modifier.isAbstract(c.getModifiers()) || Modifier.isInterface(c.getModifiers()) )
                continue;
            l.add(c);
        }

        Collections.sort(l, new Comparator<Class<?>>() {
            @Override
            public int compare(Class<?> o1, Class<?> o2) {
                return o1.getName().compareTo(o2.getName());
            }
        });

        for(Class<?> c : l){
            System.out.println(c.getName() + ".class,");
        }
    }

}
