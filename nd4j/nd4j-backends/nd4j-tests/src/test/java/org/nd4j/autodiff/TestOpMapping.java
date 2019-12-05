/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.autodiff;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.converters.ImportClassMapping;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.reflections.Reflections;

import java.lang.reflect.Modifier;
import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestOpMapping extends BaseNd4jTest {

    public TestOpMapping(Nd4jBackend b){
        super(b);
    }

    @Override
    public char ordering(){
        return 'c';
    }

    @Test
    public void testOpMappingCoverage() throws Exception {
        Reflections reflections = new Reflections("org.nd4j");
        Set<Class<? extends DifferentialFunction>> subTypes = reflections.getSubTypesOf(DifferentialFunction.class);

        Map<String, DifferentialFunction> opNameMapping = ImportClassMapping.getOpNameMapping();
        Map<String, DifferentialFunction> tfOpNameMapping = ImportClassMapping.getTFOpMappingFunctions();
        Map<String, DifferentialFunction> onnxOpNameMapping = ImportClassMapping.getOnnxOpMappingFunctions();


        for(Class<? extends DifferentialFunction> c : subTypes){

            if(Modifier.isAbstract(c.getModifiers()) || Modifier.isInterface(c.getModifiers()) || ILossFunction.class.isAssignableFrom(c))
                continue;

            DifferentialFunction df;
            try {
                df = c.newInstance();
            } catch (Throwable t){
                //All differential functions should have a no-arg constructor
                throw new RuntimeException("Error instantiating new instance - op class " + c.getName() + " likely does not have a no-arg constructor", t);
            }
            String opName = df.opName();

            assertTrue("Op is missing - not defined in ImportClassMapping: " + opName +
                    "\nInstructions to fix: Add class to org.nd4j.imports.converters.ImportClassMapping", opNameMapping.containsKey(opName)
            );

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
