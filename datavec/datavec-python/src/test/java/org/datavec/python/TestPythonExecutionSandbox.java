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

package org.datavec.python;


import org.junit.Assert;
import org.junit.Test;

@javax.annotation.concurrent.NotThreadSafe
public class TestPythonExecutionSandbox {

    @Test
    public void testInt(){
        PythonExecutioner.setInterpreter("interp1");
        PythonExecutioner.exec("a = 1");
        PythonExecutioner.setInterpreter("interp2");
        PythonExecutioner.exec("a = 2");
        PythonExecutioner.setInterpreter("interp3");
        PythonExecutioner.exec("a = 3");


        PythonExecutioner.setInterpreter("interp1");
        Assert.assertEquals(1, PythonExecutioner.evalInteger("a"));

        PythonExecutioner.setInterpreter("interp2");
        Assert.assertEquals(2, PythonExecutioner.evalInteger("a"));

        PythonExecutioner.setInterpreter("interp3");
        Assert.assertEquals(3, PythonExecutioner.evalInteger("a"));
    }

    @Test
    public void testNDArray(){
        PythonExecutioner.setInterpreter("main");
        PythonExecutioner.exec("import numpy as np");
        PythonExecutioner.exec("a = np.zeros(5)");

        PythonExecutioner.setInterpreter("main");
        //PythonExecutioner.exec("import numpy as np");
        PythonExecutioner.exec("a = np.zeros(5)");

        PythonExecutioner.setInterpreter("main");
        PythonExecutioner.exec("a += 2");

        PythonExecutioner.setInterpreter("main");
        PythonExecutioner.exec("a += 3");

        PythonExecutioner.setInterpreter("main");
        //PythonExecutioner.exec("import numpy as np");
        // PythonExecutioner.exec("a = np.zeros(5)");

        PythonExecutioner.setInterpreter("main");
        Assert.assertEquals(25, PythonExecutioner.evalNdArray("a").getNd4jArray().sum().getDouble(), 1e-5);
    }

    @Test
    public void testNumpyRandom(){
        PythonExecutioner.setInterpreter("main");
        PythonExecutioner.exec("import numpy as np; print(np.random.randint(5))");
    }
}
