/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */


import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.python4j.PythonException;
import org.nd4j.python4j.PythonGIL;
import org.nd4j.python4j.PythonObject;
import org.nd4j.python4j.PythonTypes;
import org.junit.Assert;
import org.junit.jupiter.api.Test;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import javax.annotation.concurrent.NotThreadSafe;
import java.util.*;
import java.util.stream.Stream;


@NotThreadSafe
public class PythonNumpyCollectionsTest {


    public static Stream<Arguments> params() {
        return Arrays.asList(new DataType[]{
                DataType.BOOL,
                DataType.FLOAT16,
                //DataType.BFLOAT16,
                DataType.FLOAT,
                DataType.DOUBLE,
                DataType.INT8,
                DataType.INT16,
                DataType.INT32,
                DataType.INT64,
                DataType.UINT8,
                DataType.UINT16,
                DataType.UINT32,
                DataType.UINT64
        }).stream().map(Arguments::of);
    }

    @Test
    @MethodSource("#params")
    @ParameterizedTest
    public void testPythonDictFromMap(DataType dataType) throws PythonException {
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            Map map = new HashMap();
            map.put("a", 1);
            map.put(1, "a");
            map.put("arr", Nd4j.ones(dataType, 2, 3));
            map.put("list1", Arrays.asList(1, 2.0, 3, 4f, Nd4j.zeros(dataType,3,2)));
            Map innerMap = new HashMap();
            innerMap.put("b", 2);
            innerMap.put(2, "b");
            innerMap.put(5, Nd4j.ones(dataType, 5));
            map.put("innermap", innerMap);
            map.put("list2", Arrays.asList(4, "5", innerMap, false, true));
            PythonObject dict = PythonTypes.convert(map);
            Map map2 = PythonTypes.DICT.toJava(dict);
            Assert.assertEquals(map.toString(), map2.toString());
        }

    }

    @Test
    @MethodSource("#params")
    @ParameterizedTest
    public void testPythonListFromList(DataType dataType) throws PythonException {
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            List<Object> list = new ArrayList<>();
            list.add(1);
            list.add("2");
            list.add(Nd4j.ones(dataType, 2, 3));
            list.add(Arrays.asList("a",
                    Nd4j.ones(dataType, 1, 2),1.0, 2f, 10, true, false,
                    Nd4j.zeros(dataType, 3, 2)));
            Map map = new HashMap();
            map.put("a", 1);
            map.put(1, "a");
            map.put(5, Nd4j.ones(dataType,4, 5));
            map.put("list1", Arrays.asList(1, 2.0, 3, 4f, Nd4j.zeros(dataType, 3, 1)));
            list.add(map);
            PythonObject dict = PythonTypes.convert(list);
            List list2 = PythonTypes.LIST.toJava(dict);
            Assert.assertEquals(list.toString(), list2.toString());
        }

    }
}
