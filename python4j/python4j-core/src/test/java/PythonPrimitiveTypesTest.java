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

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class PythonPrimitiveTypesTest {

    @Test
    public void testInt() throws PythonException {
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            long j = 3;
            PythonObject p = PythonTypes.INT.toPython(j);
            long j2 = PythonTypes.INT.toJava(p);

            assertEquals(j, j2);

            PythonObject p2 = PythonTypes.convert(j);
            long j3 = PythonTypes.INT.toJava(p2);

            assertEquals(j, j3);
        }

    }

    @Test
    public void testStr() throws PythonException {
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            String s = "abcd";
            PythonObject p = PythonTypes.STR.toPython(s);
            String s2 = PythonTypes.STR.toJava(p);

            assertEquals(s, s2);

            PythonObject p2 = PythonTypes.convert(s);
            String s3 = PythonTypes.STR.toJava(p2);

            assertEquals(s, s3);
        }

    }

    @Test
    public void testFloat() throws PythonException {
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            double f = 7;
            PythonObject p = PythonTypes.FLOAT.toPython(f);
            double f2 = PythonTypes.FLOAT.toJava(p);

            assertEquals(f, f2, 1e-5);

            PythonObject p2 = PythonTypes.convert(f);
            double f3 = PythonTypes.FLOAT.toJava(p2);

            assertEquals(f, f3, 1e-5);
        }

    }

    @Test
    public void testBool() throws PythonException{
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            boolean b = true;
            PythonObject p = PythonTypes.BOOL.toPython(b);
            boolean b2 = PythonTypes.BOOL.toJava(p);

            assertEquals(b, b2);

            PythonObject p2 = PythonTypes.convert(b);
            boolean b3 = PythonTypes.BOOL.toJava(p2);

            assertEquals(b, b3);
        }

    }
    @Test
    public void testBytes() {
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            byte[] bytes = new byte[256];
            for (int i = 0; i < 256; i++) {
                bytes[i] = (byte) i;
            }
            List<PythonVariable> inputs = new ArrayList<>();
            inputs.add(new PythonVariable<>("b1", PythonTypes.BYTES, bytes));
            List<PythonVariable> outputs = new ArrayList<>();
            outputs.add(new PythonVariable<>("b2", PythonTypes.BYTES));
            String code = "b2=b1";
            PythonExecutioner.exec(code, inputs, outputs);
            assertArrayEquals(bytes, (byte[]) outputs.get(0).getValue());
        }

    }

    @Test
    public void testBytes2() {
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            byte[] bytes = new byte[]{97, 98, 99};
            List<PythonVariable> inputs = new ArrayList<>();
            inputs.add(new PythonVariable<>("b1", PythonTypes.BYTES, bytes));
            List<PythonVariable> outputs = new ArrayList<>();
            outputs.add(new PythonVariable<>("s1", PythonTypes.STR));
            outputs.add(new PythonVariable<>("b2", PythonTypes.BYTES));
            String code = "s1 = ''.join(chr(c) for c in b1)\nb2=b'def'";
            PythonExecutioner.exec(code, inputs, outputs);
            assertEquals("abc", outputs.get(0).getValue());
            assertArrayEquals(new byte[]{100, 101, 102}, (byte[]) outputs.get(1).getValue());

        }
    }

}
