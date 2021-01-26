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

import org.nd4j.python4j.*;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;


@javax.annotation.concurrent.NotThreadSafe
public class PythonJobTest {

    @Test
    public void testPythonJobBasic() {
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            PythonContextManager.deleteNonMainContexts();

        }

        String code = "c = a + b";
        PythonJob job = new PythonJob("job1", code, false);

        List<PythonVariable> inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("a", PythonTypes.INT, 2));
        inputs.add(new PythonVariable<>("b", PythonTypes.INT, 3));

        List<PythonVariable> outputs = new ArrayList<>();
        outputs.add(new PythonVariable<>("c", PythonTypes.INT));


        job.exec(inputs, outputs);
        assertEquals("c", outputs.get(0).getName());
        assertEquals(5L, (long)outputs.get(0).getValue());

        inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("a", PythonTypes.FLOAT, 3.0));
        inputs.add(new PythonVariable<>("b", PythonTypes.FLOAT, 4.0));

        outputs = new ArrayList<>();
        outputs.add(new PythonVariable<>("c", PythonTypes.FLOAT));


        job.exec(inputs, outputs);

        assertEquals("c", outputs.get(0).getName());
        assertEquals(7.0, (double)outputs.get(0).getValue(), 1e-5);


    }

    @Test
    public void testPythonJobReturnAllVariables(){
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            PythonContextManager.deleteNonMainContexts();

        }

        String code = "c = a + b";
        PythonJob job = new PythonJob("job1", code, false);

        List<PythonVariable> inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("a", PythonTypes.INT, 2));
        inputs.add(new PythonVariable<>("b", PythonTypes.INT, 3));


        List<PythonVariable> outputs = job.execAndReturnAllVariables(inputs);


        assertEquals("a", outputs.get(0).getName());
        assertEquals(2L, (long)outputs.get(0).getValue());
        assertEquals("b", outputs.get(1).getName());
        assertEquals(3L, (long)outputs.get(1).getValue());
        assertEquals("c", outputs.get(2).getName());
        assertEquals(5L, (long)outputs.get(2).getValue());

        inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("a", PythonTypes.FLOAT, 3.0));
        inputs.add(new PythonVariable<>("b", PythonTypes.FLOAT, 4.0));
        outputs = job.execAndReturnAllVariables(inputs);
        assertEquals("a", outputs.get(0).getName());
        assertEquals(3.0, (double)outputs.get(0).getValue(), 1e-5);
        assertEquals("b", outputs.get(1).getName());
        assertEquals(4.0, (double)outputs.get(1).getValue(), 1e-5);
        assertEquals("c", outputs.get(2).getName());
        assertEquals(7.0, (double)outputs.get(2).getValue(), 1e-5);

    }


    @Test
    public void testMultiplePythonJobsParallel(){
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            PythonContextManager.deleteNonMainContexts();

        }
        String code1 = "c = a + b";
        PythonJob job1 = new PythonJob("job1", code1, false);

        String code2 = "c = a - b";
        PythonJob job2 = new PythonJob("job2", code2, false);

        List<PythonVariable> inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("a", PythonTypes.INT, 2));
        inputs.add(new PythonVariable<>("b", PythonTypes.INT, 3));


        List<PythonVariable> outputs = new ArrayList<>();
        outputs.add(new PythonVariable<>("c", PythonTypes.INT));

        job1.exec(inputs, outputs);

        assertEquals("c", outputs.get(0).getName());
        assertEquals(5L, (long)outputs.get(0).getValue());


        job2.exec(inputs, outputs);

        assertEquals("c", outputs.get(0).getName());
        assertEquals(-1L, (long)outputs.get(0).getValue());

        inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("a", PythonTypes.FLOAT, 3.0));
        inputs.add(new PythonVariable<>("b", PythonTypes.FLOAT, 4.0));

        outputs = new ArrayList<>();
        outputs.add(new PythonVariable<>("c", PythonTypes.FLOAT));


        job1.exec(inputs, outputs);

        assertEquals("c", outputs.get(0).getName());
        assertEquals(7.0, (double)outputs.get(0).getValue(), 1e-5);

        job2.exec(inputs, outputs);

        assertEquals("c", outputs.get(0).getName());
        assertEquals(-1., (double)outputs.get(0).getValue(), 1e-5);

    }


    @Test
    public void testPythonJobSetupRun(){
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            PythonContextManager.deleteNonMainContexts();

        }
        String code = "five=None\n" +
                "def setup():\n" +
                "    global five\n"+
                "    five = 5\n\n" +
                "def run(a, b):\n" +
                "    c = a + b + five\n"+
                "    return {'c':c}\n\n";
        PythonJob job = new PythonJob("job1", code, true);

        List<PythonVariable> inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("a", PythonTypes.INT, 2));
        inputs.add(new PythonVariable<>("b", PythonTypes.INT, 3));

        List<PythonVariable> outputs = new ArrayList<>();
        outputs.add(new PythonVariable<>("c", PythonTypes.INT));
        job.exec(inputs, outputs);

        assertEquals("c", outputs.get(0).getName());
        assertEquals(10L, (long)outputs.get(0).getValue());


        inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("a", PythonTypes.FLOAT, 3.0));
        inputs.add(new PythonVariable<>("b", PythonTypes.FLOAT, 4.0));


        outputs = new ArrayList<>();
        outputs.add(new PythonVariable<>("c", PythonTypes.FLOAT));

        job.exec(inputs, outputs);

        assertEquals("c", outputs.get(0).getName());
        assertEquals(12.0, (double)outputs.get(0).getValue(), 1e-5);

    }
    @Test
    public void testPythonJobSetupRunAndReturnAllVariables(){
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            PythonContextManager.deleteNonMainContexts();

        }
        String code = "five=None\n" +
                "c=None\n"+
                "def setup():\n" +
                "    global five\n"+
                "    five = 5\n\n" +
                "def run(a, b):\n" +
                "    global c\n" +
                "    c = a + b + five\n";
        PythonJob job = new PythonJob("job1", code, true);

        List<PythonVariable> inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("a", PythonTypes.INT, 2));
        inputs.add(new PythonVariable<>("b", PythonTypes.INT, 3));

        List<PythonVariable> outputs = job.execAndReturnAllVariables(inputs);

        assertEquals("c", outputs.get(1).getName());
        assertEquals(10L, (long)outputs.get(1).getValue());

        inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("a", PythonTypes.FLOAT, 3.0));
        inputs.add(new PythonVariable<>("b", PythonTypes.FLOAT, 4.0));

        outputs = job.execAndReturnAllVariables(inputs);


        assertEquals("c", outputs.get(1).getName());
        assertEquals(12.0, (double)outputs.get(1).getValue(), 1e-5);



    }

    @Test
    public void testMultiplePythonJobsSetupRunParallel(){
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            PythonContextManager.deleteNonMainContexts();

        }

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

        List<PythonVariable> inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("a", PythonTypes.INT, 2));
        inputs.add(new PythonVariable<>("b", PythonTypes.INT, 3));


        List<PythonVariable> outputs = new ArrayList<>();
        outputs.add(new PythonVariable<>("c", PythonTypes.INT));

        job1.exec(inputs, outputs);

        assertEquals("c", outputs.get(0).getName());
        assertEquals(10L, (long)outputs.get(0).getValue());

        job2.exec(inputs, outputs);

        assertEquals("c", outputs.get(0).getName());
        assertEquals(0L, (long)outputs.get(0).getValue());

        inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("a", PythonTypes.FLOAT, 3.0));
        inputs.add(new PythonVariable<>("b", PythonTypes.FLOAT, 4.0));

        outputs = new ArrayList<>();
        outputs.add(new PythonVariable<>("c", PythonTypes.FLOAT));


        job1.exec(inputs, outputs);

        assertEquals("c", outputs.get(0).getName());
        assertEquals(12.0, (double)outputs.get(0).getValue(), 1e-5);

        job2.exec(inputs, outputs);

        assertEquals("c", outputs.get(0).getName());
        assertEquals(2.0, (double)outputs.get(0).getValue(), 1e-5);

    }

}
