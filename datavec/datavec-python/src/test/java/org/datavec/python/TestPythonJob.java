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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;


@javax.annotation.concurrent.NotThreadSafe
public class TestPythonJob {

    @Test
    public void testPythonJobBasic() throws Exception{
        PythonContextManager.deleteNonMainContexts();

        String code = "c = a + b";
        PythonJob job = new PythonJob("job1", code, false);

        PythonVariables inputs = new PythonVariables();
        inputs.addInt("a", 2);
        inputs.addInt("b", 3);

        PythonVariables outputs = new PythonVariables();
        outputs.addInt("c");

        job.exec(inputs, outputs);

        assertEquals(5L, (long)outputs.getIntValue("c"));

        inputs = new PythonVariables();
        inputs.addFloat("a", 3.0);
        inputs.addFloat("b", 4.0);

        outputs = new PythonVariables();
        outputs.addFloat("c");


        job.exec(inputs, outputs);

        assertEquals(7.0, outputs.getFloatValue("c"), 1e-5);


        inputs = new PythonVariables();
        inputs.addNDArray("a", Nd4j.zeros(3, 2).add(4));
        inputs.addNDArray("b", Nd4j.zeros(3, 2).add(5));

        outputs = new PythonVariables();
        outputs.addNDArray("c");


        job.exec(inputs, outputs);

        assertEquals(Nd4j.zeros(3, 2).add(9), outputs.getNDArrayValue("c"));
    }

    @Test
    public void testPythonJobReturnAllVariables()throws Exception{
        PythonContextManager.deleteNonMainContexts();

        String code = "c = a + b";
        PythonJob job = new PythonJob("job1", code, false);

        PythonVariables inputs = new PythonVariables();
        inputs.addInt("a", 2);
        inputs.addInt("b", 3);


        PythonVariables outputs = job.execAndReturnAllVariables(inputs);

        assertEquals(5L, (long)outputs.getIntValue("c"));

        inputs = new PythonVariables();
        inputs.addFloat("a", 3.0);
        inputs.addFloat("b", 4.0);

        outputs = job.execAndReturnAllVariables(inputs);

        assertEquals(7.0, outputs.getFloatValue("c"), 1e-5);


        inputs = new PythonVariables();
        inputs.addNDArray("a", Nd4j.zeros(3, 2).add(4));
        inputs.addNDArray("b", Nd4j.zeros(3, 2).add(5));

        outputs = job.execAndReturnAllVariables(inputs);

        assertEquals(Nd4j.zeros(3, 2).add(9), outputs.getNDArrayValue("c"));
    }

    @Test
    public void testMultiplePythonJobsParallel()throws Exception{
        PythonContextManager.deleteNonMainContexts();

        String code1 = "c = a + b";
        PythonJob job1 = new PythonJob("job1", code1, false);

        String code2 = "c = a - b";
        PythonJob job2 = new PythonJob("job2", code2, false);

        PythonVariables inputs = new PythonVariables();
        inputs.addInt("a", 2);
        inputs.addInt("b", 3);

        PythonVariables outputs = new PythonVariables();
        outputs.addInt("c");

        job1.exec(inputs, outputs);

        assertEquals(5L, (long)outputs.getIntValue("c"));

        job2.exec(inputs, outputs);

        assertEquals(-1L, (long)outputs.getIntValue("c"));

        inputs = new PythonVariables();
        inputs.addFloat("a", 3.0);
        inputs.addFloat("b", 4.0);

        outputs = new PythonVariables();
        outputs.addFloat("c");


        job1.exec(inputs, outputs);

        assertEquals(7.0, outputs.getFloatValue("c"), 1e-5);

        job2.exec(inputs, outputs);

        assertEquals(-1L, outputs.getFloatValue("c"), 1e-5);


        inputs = new PythonVariables();
        inputs.addNDArray("a", Nd4j.zeros(3, 2).add(4));
        inputs.addNDArray("b", Nd4j.zeros(3, 2).add(5));

        outputs = new PythonVariables();
        outputs.addNDArray("c");


        job1.exec(inputs, outputs);

        assertEquals(Nd4j.zeros(3, 2).add(9), outputs.getNDArrayValue("c"));

        job2.exec(inputs, outputs);

        assertEquals(Nd4j.zeros(3, 2).sub(1), outputs.getNDArrayValue("c"));
    }
    @Test
    public void testPythonJobSetupRun()throws Exception{
        PythonContextManager.deleteNonMainContexts();

        String code = "five=None\n" +
                "def setup():\n" +
                "    global five\n"+
                "    five = 5\n\n" +
                "def run(a, b):\n" +
                "    c = a + b + five\n"+
                "    return {'c':c}\n\n";
        PythonJob job = new PythonJob("job1", code, true);

        PythonVariables inputs = new PythonVariables();
        inputs.addInt("a", 2);
        inputs.addInt("b", 3);

        PythonVariables outputs = new PythonVariables();
        outputs.addInt("c");

        job.exec(inputs, outputs);

        assertEquals(10L, (long)outputs.getIntValue("c"));

        inputs = new PythonVariables();
        inputs.addFloat("a", 3.0);
        inputs.addFloat("b", 4.0);

        outputs = new PythonVariables();
        outputs.addFloat("c");


        job.exec(inputs, outputs);

        assertEquals(12.0, outputs.getFloatValue("c"), 1e-5);


        inputs = new PythonVariables();
        inputs.addNDArray("a", Nd4j.zeros(3, 2).add(4));
        inputs.addNDArray("b", Nd4j.zeros(3, 2).add(5));

        outputs = new PythonVariables();
        outputs.addNDArray("c");


        job.exec(inputs, outputs);

        assertEquals(Nd4j.zeros(3, 2).add(14), outputs.getNDArrayValue("c"));
    }
    @Test
    public void testPythonJobSetupRunAndReturnAllVariables()throws Exception{
        PythonContextManager.deleteNonMainContexts();

        String code = "five=None\n" +
                "def setup():\n" +
                "    global five\n"+
                "    five = 5\n\n" +
                "def run(a, b):\n" +
                "    c = a + b + five\n"+
                "    return {'c':c}\n\n";
        PythonJob job = new PythonJob("job1", code, true);

        PythonVariables inputs = new PythonVariables();
        inputs.addInt("a", 2);
        inputs.addInt("b", 3);


        PythonVariables outputs = job.execAndReturnAllVariables(inputs);

        assertEquals(10L, (long)outputs.getIntValue("c"));

        inputs = new PythonVariables();
        inputs.addFloat("a", 3.0);
        inputs.addFloat("b", 4.0);

        outputs = job.execAndReturnAllVariables(inputs);

        assertEquals(12.0, outputs.getFloatValue("c"), 1e-5);


        inputs = new PythonVariables();
        inputs.addNDArray("a", Nd4j.zeros(3, 2).add(4));
        inputs.addNDArray("b", Nd4j.zeros(3, 2).add(5));

        outputs = job.execAndReturnAllVariables(inputs);

        assertEquals(Nd4j.zeros(3, 2).add(14), outputs.getNDArrayValue("c"));
    }

    @Test
    public void testMultiplePythonJobsSetupRunParallel()throws Exception{
        PythonContextManager.deleteNonMainContexts();

        String code1 = "five=None\n" +
                "def setup():\n" +
                "    global five\n"+
                "    five = 5\n\n" +
                "def run(a, b):\n" +
                "    c = a + b + five\n"+
                "    return {'c':c}\n\n";
        PythonJob job1 = new PythonJob("job1", code1, true);

        String code2 = "five=None\n" +
                "def setup():\n" +
                "    global five\n"+
                "    five = 5\n\n" +
                "def run(a, b):\n" +
                "    c = a + b - five\n"+
                "    return {'c':c}\n\n";
        PythonJob job2 = new PythonJob("job2", code2, true);

        PythonVariables inputs = new PythonVariables();
        inputs.addInt("a", 2);
        inputs.addInt("b", 3);

        PythonVariables outputs = new PythonVariables();
        outputs.addInt("c");

        job1.exec(inputs, outputs);

        assertEquals(10L, (long)outputs.getIntValue("c"));

        job2.exec(inputs, outputs);

        assertEquals(0L, (long)outputs.getIntValue("c"));

        inputs = new PythonVariables();
        inputs.addFloat("a", 3.0);
        inputs.addFloat("b", 4.0);

        outputs = new PythonVariables();
        outputs.addFloat("c");


        job1.exec(inputs, outputs);

        assertEquals(12.0, outputs.getFloatValue("c"), 1e-5);

        job2.exec(inputs, outputs);

        assertEquals(2L, outputs.getFloatValue("c"), 1e-5);


        inputs = new PythonVariables();
        inputs.addNDArray("a", Nd4j.zeros(3, 2).add(4));
        inputs.addNDArray("b", Nd4j.zeros(3, 2).add(5));

        outputs = new PythonVariables();
        outputs.addNDArray("c");


        job1.exec(inputs, outputs);

        assertEquals(Nd4j.zeros(3, 2).add(14), outputs.getNDArrayValue("c"));

        job2.exec(inputs, outputs);

        assertEquals(Nd4j.zeros(3, 2).add(4), outputs.getNDArrayValue("c"));
    }

}
