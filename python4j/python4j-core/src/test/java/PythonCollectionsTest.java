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


import org.nd4j.python4j.*;

import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.assertEquals;


@javax.annotation.concurrent.NotThreadSafe
public class PythonCollectionsTest {


    @Test
    public void testPythonDictFromMap() throws PythonException {
      try(PythonGIL pythonGIL = PythonGIL.lock()) {
          Map map = new HashMap();
          map.put("a", 1);
          map.put(1, "a");
          map.put("list1", Arrays.asList(1, 2.0, 3, 4f));
          Map innerMap = new HashMap();
          innerMap.put("b", 2);
          innerMap.put(2, "b");
          map.put("innermap", innerMap);
          map.put("list2", Arrays.asList(4, "5", innerMap, false, true));
          PythonObject dict = PythonTypes.convert(map);
          Map map2 = PythonTypes.DICT.toJava(dict);
          assertEquals(map.toString(), map2.toString());
      }

    }

    @Test
    public void testPythonListFromList() throws PythonException{
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            List<Object> list = new ArrayList<>();
            list.add(1);
            list.add("2");
            list.add(Arrays.asList("a", 1.0, 2f, 10, true, false));
            Map map = new HashMap();
            map.put("a", 1);
            map.put(1, "a");
            map.put("list1", Arrays.asList(1, 2.0, 3, 4f));
            list.add(map);
            PythonObject dict = PythonTypes.convert(list);
            List list2 = PythonTypes.LIST.toJava(dict);
            assertEquals(list.toString(), list2.toString());
        }

    }
}
