
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


import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import javax.annotation.concurrent.NotThreadSafe;

@NotThreadSafe
public class TestPythonContextManager {

    @Test
    public void testInt() throws Exception{
        Python.setContext("context1");
        Python.exec("a = 1");
        Python.setContext("context2");
        Python.exec("a = 2");
        Python.setContext("context3");
        Python.exec("a = 3");


        Python.setContext("context1");
        Assert.assertEquals(1, PythonExecutioner.getVariable("a").toInt());

        Python.setContext("context2");
        Assert.assertEquals(2, PythonExecutioner.getVariable("a").toInt());

        Python.setContext("context3");
        Assert.assertEquals(3, PythonExecutioner.getVariable("a").toInt());

        PythonContextManager.deleteNonMainContexts();
    }

    @Test
    public void testNDArray() throws Exception{
        Python.setContext("context1");
        Python.exec("import numpy as np");
        Python.exec("a = np.zeros((3,2)) + 1");

        Python.setContext("context2");
        Python.exec("import numpy as np");
        Python.exec("a = np.zeros((3,2)) + 2");

        Python.setContext("context3");
        Python.exec("import numpy as np");
        Python.exec("a = np.zeros((3,2)) + 3");

        Python.setContext("context1");
        Python.exec("a += 1");

        Python.setContext("context2");
        Python.exec("a += 2");

        Python.setContext("context3");
        Python.exec("a += 3");

        INDArray arr = Nd4j.create(DataType.DOUBLE, 3, 2);
        Python.setContext("context1");
        Assert.assertEquals(arr.add(2), PythonExecutioner.getVariable("a").toNumpy().getNd4jArray());

        Python.setContext("context2");
        Assert.assertEquals(arr.add(4), PythonExecutioner.getVariable("a").toNumpy().getNd4jArray());

        Python.setContext("context3");
        Assert.assertEquals(arr.add(6), PythonExecutioner.getVariable("a").toNumpy().getNd4jArray());
    }

}
