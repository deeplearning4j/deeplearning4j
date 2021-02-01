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

import org.bytedeco.javacpp.BytePointer;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static junit.framework.TestCase.assertNotNull;
import static junit.framework.TestCase.assertNull;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestPythonVariables {

    @Test
    public void testDataAssociations() throws PythonException{
        PythonVariables pythonVariables = new PythonVariables();
        PythonType[] types = {
                PythonType.INT,
                PythonType.FLOAT,
                PythonType.STR,
                PythonType.BOOL,
                PythonType.DICT,
                PythonType.LIST,
                PythonType.LIST,
                PythonType.NDARRAY,
                PythonType.BYTES
        };

        INDArray arr = Nd4j.scalar(1.0);
        ((BaseDataBuffer)arr.data()).syncToPrimary();
        BytePointer bp = new BytePointer(arr.data().pointer());
        Object[] values = {
                1L,1.0,"1",true, Collections.singletonMap("1",1),
                new Object[]{1}, Arrays.asList(1), arr, bp
        };

        Object[] expectedValues = {
                1L,1.0,"1",true, Collections.singletonMap("1",1),
                Arrays.asList(1), Arrays.asList(1), arr, bp
        };

        for(int i = 0; i < types.length; i++) {
            testInsertGet(pythonVariables,types[i].getName().name() + i,values[i],types[i],expectedValues[i]);
        }

        assertEquals(types.length,pythonVariables.getVariables().length);

    }

    private void testInsertGet(PythonVariables pythonVariables,String key,Object value,PythonType type,Object expectedValue) throws PythonException{
        pythonVariables.add(key, type);
        assertNull(pythonVariables.getValue(key));
        pythonVariables.setValue(key,value);
        assertNotNull(pythonVariables.getValue(key));
        Object actualValue = pythonVariables.getValue(key);
        if (expectedValue instanceof Object[]){
            assertTrue(actualValue instanceof List);
            Object[] actualArr = ((List)actualValue).toArray();
            Object[] expectedArr = (Object[])expectedValue;
            assertArrayEquals(expectedArr, actualArr);
        }
        else{
            assertEquals(expectedValue,pythonVariables.getValue(key));
        }

    }


}
