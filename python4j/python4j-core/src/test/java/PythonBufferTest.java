/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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


import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.eclipse.python4j.*;
import org.junit.Assert;
import org.junit.Test;
import sun.nio.ch.DirectBuffer;

import javax.annotation.concurrent.NotThreadSafe;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.*;

@NotThreadSafe
public class PythonBufferTest {

    @Test
    public void testBuffer() {
        ByteBuffer buff = ByteBuffer.allocateDirect(3);
        buff.put((byte) 97);
        buff.put((byte) 98);
        buff.put((byte) 99);
        buff.rewind();

        BytePointer bp = new BytePointer(buff);

        List<PythonVariable> inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("buff", PythonTypes.MEMORYVIEW, buff));

        List<PythonVariable> outputs = new ArrayList<>();
        outputs.add(new PythonVariable<>("s1", PythonTypes.STR));
        outputs.add(new PythonVariable<>("s2", PythonTypes.STR));

        String code = "s1 = ''.join(chr(c) for c in buff)\nbuff[2] += 2\ns2 = ''.join(chr(c) for c in buff)";

        PythonExecutioner.exec(code, inputs, outputs);
        Assert.assertEquals("abc", outputs.get(0).getValue());
        Assert.assertEquals("abe", outputs.get(1).getValue());
        Assert.assertEquals(101, buff.get(2));

    }
    @Test
    public void testBuffer2() {
        ByteBuffer buff = ByteBuffer.allocateDirect(3);
        buff.put((byte) 97);
        buff.put((byte) 98);
        buff.put((byte) 99);
        buff.rewind();

        BytePointer bp = new BytePointer(buff);

        List<PythonVariable> inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("buff", PythonTypes.MEMORYVIEW, bp));

        List<PythonVariable> outputs = new ArrayList<>();
        outputs.add(new PythonVariable<>("s1", PythonTypes.STR));
        outputs.add(new PythonVariable<>("s2", PythonTypes.STR));

        String code = "s1 = ''.join(chr(c) for c in buff)\nbuff[2] += 2\ns2 = ''.join(chr(c) for c in buff)";

        PythonExecutioner.exec(code, inputs, outputs);
        Assert.assertEquals("abc", outputs.get(0).getValue());
        Assert.assertEquals("abe", outputs.get(1).getValue());
        Assert.assertEquals(101, buff.get(2));

    }

    @Test
    public void testBuffer3() {
        ByteBuffer buff = ByteBuffer.allocateDirect(3);
        buff.put((byte) 97);
        buff.put((byte) 98);
        buff.put((byte) 99);
        buff.rewind();

        BytePointer bp = new BytePointer(buff);

        List<PythonVariable> inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("buff", PythonTypes.MEMORYVIEW, bp));

        List<PythonVariable> outputs = new ArrayList<>();
        outputs.add(new PythonVariable<>("s1", PythonTypes.STR));
        outputs.add(new PythonVariable<>("s2", PythonTypes.STR));
        outputs.add(new PythonVariable<>("buff2", PythonTypes.MEMORYVIEW));
        String code = "s1 = ''.join(chr(c) for c in buff)\nbuff[2] += 2\ns2 = ''.join(chr(c) for c in buff)\nbuff2=buff[1:]";
        PythonExecutioner.exec(code, inputs, outputs);

        Assert.assertEquals("abc", outputs.get(0).getValue());
        Assert.assertEquals("abe", outputs.get(1).getValue());
        Assert.assertEquals(101, buff.get(2));
        BytePointer outBuffer = (BytePointer) outputs.get(2).getValue();
        Assert.assertEquals(2, outBuffer.capacity());
        Assert.assertEquals((byte)98, outBuffer.get(0));
        Assert.assertEquals((byte)101, outBuffer.get(1));

    }
}