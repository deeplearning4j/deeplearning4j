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


import org.junit.Assert;
import org.junit.Test;
import org.nd4j.python4j.*;

import javax.annotation.concurrent.NotThreadSafe;
import java.util.*;

@NotThreadSafe
public class PythonBasicExecutionTest {

    @Test(expected =  IllegalStateException.class)
    public void testSimpleExecIllegal() {
        String code = "print('Hello World')";
        PythonExecutioner.exec(code);

    }

    @Test
    public void testSimpleExec() {
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            String code = "print('Hello World')";
            PythonExecutioner.exec(code);
        }

    }

    @Test
    public void testBadCode() throws Exception {
        try {
            try(PythonGIL pythonGIL = PythonGIL.lock()) {
                String code = "printx('Hello world')";
                PythonExecutioner.exec(code);
            }

        } catch (Exception e) {
            Assert.assertEquals("NameError: name 'printx' is not defined", e.getMessage());
            return;
        }
        throw new Exception("Bad code did not throw!");
    }

    @Test
    public void testExecWithInputs() {
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            List<PythonVariable> inputs = new ArrayList<>();
            inputs.add(new PythonVariable<>("x", PythonTypes.STR, "Hello "));
            inputs.add(new PythonVariable<>("y", PythonTypes.STR, "World"));
            String code = "print(x + y)";
            PythonExecutioner.exec(code, inputs, null);
        }

    }

    @Test
    public void testExecWithInputsAndOutputs() {
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            List<PythonVariable> inputs = new ArrayList<>();
            inputs.add(new PythonVariable<>("x", PythonTypes.STR, "Hello "));
            inputs.add(new PythonVariable<>("y", PythonTypes.STR, "World"));
            PythonVariable out = new PythonVariable<>("z", PythonTypes.STR);
            String code = "z = x + y";
            PythonExecutioner.exec(code, inputs, Collections.singletonList(out));
            Assert.assertEquals("Hello World", out.getValue());
        }
    }

    @Test
    public void testExecAndReturnAllVariables() {
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            PythonContextManager.reset();
            String code = "a = 5\nb = '10'\nc = 20.0";
            List<PythonVariable> vars = PythonExecutioner.execAndReturnAllVariables(code);

            Assert.assertEquals("a", vars.get(0).getName());
            Assert.assertEquals(PythonTypes.INT, vars.get(0).getType());
            Assert.assertEquals(5L, (long) vars.get(0).getValue());

            Assert.assertEquals("b", vars.get(1).getName());
            Assert.assertEquals(PythonTypes.STR, vars.get(1).getType());
            Assert.assertEquals("10", vars.get(1).getValue().toString());

            Assert.assertEquals("c", vars.get(2).getName());
            Assert.assertEquals(PythonTypes.FLOAT, vars.get(2).getType());
            Assert.assertEquals(20.0, (double) vars.get(2).getValue(), 1e-5);

        }
    }

    @Test
    public void testExecWithInputsAndReturnAllVariables() {
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            PythonContextManager.reset();
            List<PythonVariable> inputs = new ArrayList<>();
            inputs.add(new PythonVariable<>("a", PythonTypes.INT, 5));
            String code = "b = '10'\nc = 20.0 + a";
            List<PythonVariable> vars = PythonExecutioner.execAndReturnAllVariables(code, inputs);

            Assert.assertEquals("a", vars.get(0).getName());
            Assert.assertEquals(PythonTypes.INT, vars.get(0).getType());
            Assert.assertEquals(5L, (long) vars.get(0).getValue());

            Assert.assertEquals("b", vars.get(1).getName());
            Assert.assertEquals(PythonTypes.STR, vars.get(1).getType());
            Assert.assertEquals("10", vars.get(1).getValue().toString());

            Assert.assertEquals("c", vars.get(2).getName());
            Assert.assertEquals(PythonTypes.FLOAT, vars.get(2).getType());
            Assert.assertEquals(25.0, (double) vars.get(2).getValue(), 1e-5);

        }
    }

}
