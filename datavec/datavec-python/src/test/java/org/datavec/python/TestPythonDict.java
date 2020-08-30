
/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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

package org.datavec.python;


import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

@javax.annotation.concurrent.NotThreadSafe
public class TestPythonDict {

    @Test
    public void testPythonDictFromMap() throws Exception{
        Map<Object, Object> map = new HashMap<>();
        map.put("a", 1);
        map.put("b", "a");
        map.put("1", Arrays.asList(1, 2, 3, "4", Arrays.asList("x", 2.3)));
        Map<Object, Object> innerMap = new HashMap<>();
        innerMap.put("k", 32);
        map.put("inner", innerMap);
        map.put("ndarray", Nd4j.linspace(1, 4, 4));
        innerMap.put("ndarray", Nd4j.linspace(5, 8, 4));
        PythonObject dict = new PythonObject(map);
        assertEquals(map.size(), Python.len(dict).toInt());
        assertEquals("{'a': 1, '1': [1, 2, 3, '4', ['" +
                        "x', 2.3]], 'b': 'a', 'inner': {'k': 32," +
                        " 'ndarray': array([5., 6., 7., 8.], dty" +
                        "pe=float32)}, 'ndarray': array([1., 2., " +
                        "3., 4.], dtype=float32)}",
                dict.toString());
        Map map2 = dict.toMap();
        PythonObject dict2 = new PythonObject(map2);
        assertEquals(dict.toString(), dict2.toString());


    }

}
