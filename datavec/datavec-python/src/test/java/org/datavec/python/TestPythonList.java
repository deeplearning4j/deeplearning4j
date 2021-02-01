
/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.python;


import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.Assert.assertEquals;

@javax.annotation.concurrent.NotThreadSafe
public class TestPythonList {

    @Test
    public void testPythonListFromIntArray() {
        PythonObject pyList = new PythonObject(new Integer[]{1, 2, 3, 4, 5});
        pyList.attr("append").call(6);
        pyList.attr("append").call(7);
        pyList.attr("append").call(8);
        assertEquals(8, Python.len(pyList).toInt());
        for (int i = 0; i < 8; i++) {
            assertEquals(i + 1, pyList.get(i).toInt());
        }

    }

    @Test
    public void testPythonListFromLongArray() {
        PythonObject pyList = new PythonObject(new Long[]{1L, 2L, 3L, 4L, 5L});
        pyList.attr("append").call(6);
        pyList.attr("append").call(7);
        pyList.attr("append").call(8);
        assertEquals(8, Python.len(pyList).toInt());
        for (int i = 0; i < 8; i++) {
            assertEquals(i + 1, pyList.get(i).toInt());
        }

    }

    @Test
    public void testPythonListFromDoubleArray() {
        PythonObject pyList = new PythonObject(new Double[]{1., 2., 3., 4., 5.});
        pyList.attr("append").call(6);
        pyList.attr("append").call(7);
        pyList.attr("append").call(8);
        assertEquals(8, Python.len(pyList).toInt());
        for (int i = 0; i < 8; i++) {
            assertEquals(i + 1, pyList.get(i).toInt());
            assertEquals((double) i + 1, pyList.get(i).toDouble(), 1e-5);
        }

    }

    @Test
    public void testPythonListFromStringArray() {
        PythonObject pyList = new PythonObject(new String[]{"abcd", "efg"});
        pyList.attr("append").call("hijk");
        pyList.attr("append").call("lmnop");
        assertEquals("abcdefghijklmnop", new PythonObject("").attr("join").call(pyList).toString());
    }

    @Test
    public void testPythonListFromMixedArray()throws Exception {
        Map<Object, Object> map = new HashMap<>();
        map.put(1, "a");
        map.put("a", Arrays.asList("a", "b", "c"));
        map.put("arr", Nd4j.linspace(1, 4, 4));
        Object[] objs = new Object[]{
                1, 2, "a", 3f, 4L, 5.0, Arrays.asList(10,
                20, "b", 30f, 40L, 50.0, map

        ), map
        };
        PythonObject pyList = new PythonObject(objs);
        System.out.println(pyList.toString());
        String expectedStr = "[1, 2, 'a', 3.0, 4, 5.0, [10" +
                ", 20, 'b', 30.0, 40, 50.0, {'arr': array([1.," +
                " 2., 3., 4.], dtype=float32), 1: 'a', 'a': [" +
                "'a', 'b', 'c']}], {'arr': array([1., 2., 3.," +
                " 4.], dtype=float32), 1: 'a', 'a': ['a', 'b', 'c']}]";
        assertEquals(expectedStr, pyList.toString());
        List objs2 = pyList.toList();
        PythonObject pyList2 = new PythonObject(objs2);
        assertEquals(pyList.toString(), pyList2.toString());
    }

}
