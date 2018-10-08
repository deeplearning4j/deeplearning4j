/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.arrow;

import com.google.flatbuffers.FlatBufferBuilder;
import org.apache.arrow.flatbuf.Tensor;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.reflect.Method;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;

public class ArrowSerdeTest {

    @BeforeClass
    public static void before(){
        Class<?> c = FlatBufferBuilder.class;
        ClassLoader cl = ArrowSerdeTest.class.getClassLoader();
        System.out.println("FlatBufferBuilder location: " + cl.getResource("com/google/flatbuffers/FlatBufferBuilder.class"));
        Method[] methods = c.getDeclaredMethods();
        System.out.println("FlatBufferBuilder Methods:");
        for(Method m : methods){
            Class<?>[] paramTypes = m.getParameterTypes();
            System.out.print("  - " + m.getName() + "(");
            boolean first = true;
            for(Class<?> p : paramTypes){
                if(!first){
                    System.out.print(",");
                }
                System.out.print(p.getSimpleName());
                first = false;
            }
            System.out.println(")");
        }
    }

    @Test
    public void testBackAndForth() {
        INDArray arr = Nd4j.linspace(1,4,4);
        Tensor tensor = ArrowSerde.toTensor(arr);
        INDArray arr2 = ArrowSerde.fromTensor(tensor);
        assertEquals(arr,arr2);
    }


    @Test
    public void testSerializeView() {
        INDArray matrix = Nd4j.linspace(1,8,8).reshape(2,4);
        Tensor tensor = ArrowSerde.toTensor(matrix.slice(0));
        INDArray from = ArrowSerde.fromTensor(tensor);
        assertEquals(matrix.data().dataType(),from.data().dataType());
        assertEquals(matrix.slice(0),from);
    }

}
