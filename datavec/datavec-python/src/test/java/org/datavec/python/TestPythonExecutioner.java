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
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.lang.reflect.Method;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;


@javax.annotation.concurrent.NotThreadSafe
public class TestPythonExecutioner {


    @org.junit.Test
    public void testPythonSysVersion() throws PythonException {
        Python.exec("import sys; print(sys.version)");
    }

    @Test
    public void testStr() throws Exception {

        PythonVariables pyInputs = new PythonVariables();
        PythonVariables pyOutputs = new PythonVariables();

        pyInputs.addStr("x", "Hello");
        pyInputs.addStr("y", "World");

        pyOutputs.addStr("z");

        String code = "z = x + ' ' + y";

        Python.exec(code, pyInputs, pyOutputs);

        String z = pyOutputs.getStrValue("z");

        System.out.println(z);

        assertEquals("Hello World", z);
    }

    @Test
    public void testInt() throws Exception {
        PythonVariables pyInputs = new PythonVariables();
        PythonVariables pyOutputs = new PythonVariables();

        pyInputs.addInt("x", 10);
        pyInputs.addInt("y", 20);

        String code = "z = x + y";

        pyOutputs.addInt("z");


        Python.exec(code, pyInputs, pyOutputs);

        long z = pyOutputs.getIntValue("z");

        Assert.assertEquals(30, z);

    }

    @Test
    public void testList() throws Exception {
        PythonVariables pyInputs = new PythonVariables();
        PythonVariables pyOutputs = new PythonVariables();

        Object[] x = new Object[]{1L, 2L, 3L, "a", "b", "c"};
        Object[] y = new Object[]{4L, 5L, 6L, "d", "e", "f"};

        pyInputs.addList("x", x);
        pyInputs.addList("y", y);

        String code = "z = x + y";

        pyOutputs.addList("z");


        Python.exec(code, pyInputs, pyOutputs);

        Object[] z = pyOutputs.getListValue("z").toArray();

        Assert.assertEquals(z.length, x.length + y.length);

        for (int i = 0; i < x.length; i++) {
            if (x[i] instanceof Number) {
                Number xNum = (Number) x[i];
                Number zNum = (Number) z[i];
                Assert.assertEquals(xNum.intValue(), zNum.intValue());
            } else {
                Assert.assertEquals(x[i], z[i]);
            }

        }
        for (int i = 0; i < y.length; i++) {
            if (y[i] instanceof Number) {
                Number yNum = (Number) y[i];
                Number zNum = (Number) z[x.length + i];
                Assert.assertEquals(yNum.intValue(), zNum.intValue());
            } else {
                Assert.assertEquals(y[i], z[x.length + i]);

            }

        }

    }

    @Test
    public void testNDArrayFloat() throws Exception {
        PythonVariables pyInputs = new PythonVariables();
        PythonVariables pyOutputs = new PythonVariables();

        pyInputs.addNDArray("x", Nd4j.zeros(DataType.FLOAT, 2, 3));
        pyInputs.addNDArray("y", Nd4j.ones(DataType.FLOAT, 2, 3));
        pyOutputs.addNDArray("z");

        String code = "z = x + y";

        Python.exec(code, pyInputs, pyOutputs);
        INDArray z = pyOutputs.getNDArrayValue("z");

        Assert.assertEquals(6.0, z.sum().getDouble(0), 1e-5);


    }

    @Test
    @Ignore
    public void testTensorflowCustomAnaconda() throws PythonException {
        Python.exec("import tensorflow as tf");
    }

    @Test
    public void testNDArrayDouble() throws Exception {
        PythonVariables pyInputs = new PythonVariables();
        PythonVariables pyOutputs = new PythonVariables();

        pyInputs.addNDArray("x", Nd4j.zeros(DataType.DOUBLE, 2, 3));
        pyInputs.addNDArray("y", Nd4j.ones(DataType.DOUBLE, 2, 3));
        pyOutputs.addNDArray("z");

        String code = "z = x + y";

        Python.exec(code, pyInputs, pyOutputs);
        INDArray z = pyOutputs.getNDArrayValue("z");

        Assert.assertEquals(6.0, z.sum().getDouble(0), 1e-5);
    }

    @Test
    public void testNDArrayShort() throws Exception {
        PythonVariables pyInputs = new PythonVariables();
        PythonVariables pyOutputs = new PythonVariables();

        pyInputs.addNDArray("x", Nd4j.zeros(DataType.SHORT, 2, 3));
        pyInputs.addNDArray("y", Nd4j.ones(DataType.SHORT, 2, 3));
        pyOutputs.addNDArray("z");

        String code = "z = x + y";

        Python.exec(code, pyInputs, pyOutputs);
        INDArray z = pyOutputs.getNDArrayValue("z");

        Assert.assertEquals(6.0, z.sum().getDouble(0), 1e-5);
    }


    @Test
    public void testNDArrayInt() throws Exception {
        PythonVariables pyInputs = new PythonVariables();
        PythonVariables pyOutputs = new PythonVariables();

        pyInputs.addNDArray("x", Nd4j.zeros(DataType.INT, 2, 3));
        pyInputs.addNDArray("y", Nd4j.ones(DataType.INT, 2, 3));
        pyOutputs.addNDArray("z");

        String code = "z = x + y";

        Python.exec(code, pyInputs, pyOutputs);
        INDArray z = pyOutputs.getNDArrayValue("z");

        Assert.assertEquals(6.0, z.sum().getDouble(0), 1e-5);

    }

    @Test
    public void testNDArrayLong() throws Exception {
        PythonVariables pyInputs = new PythonVariables();
        PythonVariables pyOutputs = new PythonVariables();

        pyInputs.addNDArray("x", Nd4j.zeros(DataType.LONG, 2, 3));
        pyInputs.addNDArray("y", Nd4j.ones(DataType.LONG, 2, 3));
        pyOutputs.addNDArray("z");

        String code = "z = x + y";

        Python.exec(code, pyInputs, pyOutputs);
        INDArray z = pyOutputs.getNDArrayValue("z");

        Assert.assertEquals(6.0, z.sum().getDouble(0), 1e-5);

    }


    @Test
    public void testNDArrayNoCopy() throws Exception{
        PythonVariables pyInputs = new PythonVariables();
        PythonVariables pyOutputs = new PythonVariables();
        INDArray arr = Nd4j.rand(3, 2);
        ((BaseDataBuffer)arr.data()).syncToPrimary();
        pyInputs.addNDArray("x", arr);
        pyOutputs.addNDArray("x");
        INDArray expected = arr.mul(2.3);
        String code = "x *= 2.3";
        Python.exec(code, pyInputs, pyOutputs);
        Assert.assertEquals(pyInputs.getNDArrayValue("x"), pyOutputs.getNDArrayValue("x"));
        Assert.assertEquals(expected, pyOutputs.getNDArrayValue("x"));
        Assert.assertEquals(arr.data().address(), pyOutputs.getNDArrayValue("x").data().address());
    }

    @Test
    public void testNDArrayInplace() throws Exception{
        PythonVariables pyInputs = new PythonVariables();
        INDArray arr = Nd4j.rand(3, 2);
        ((BaseDataBuffer)arr.data()).syncToPrimary();
        pyInputs.addNDArray("x", arr);
        INDArray expected = arr.mul(2.3);
        String code = "x *= 2.3";
        Python.exec(code, pyInputs, null);
        Assert.assertEquals(expected, arr);
    }

    @Test
    public void testByteBufferInput() throws Exception{
        //ByteBuffer buff = ByteBuffer.allocateDirect(3);
        INDArray buff = Nd4j.zeros(new int[]{3}, DataType.BYTE);
        buff.putScalar(0, 97); // a
        buff.putScalar(1, 98); // b
        buff.putScalar(2, 99); // c
        ((BaseDataBuffer)buff.data()).syncToPrimary();
        PythonVariables pyInputs = new PythonVariables();
        pyInputs.addBytes("buff", new BytePointer(buff.data().pointer()));

        PythonVariables pyOutputs= new PythonVariables();
        pyOutputs.addStr("out");

        String code = "out = bytes(buff).decode()";
        Python.exec(code, pyInputs, pyOutputs);
        Assert.assertEquals("abc", pyOutputs.getStrValue("out"));

      }


      @Test
      public void testByteBufferOutputNoCopy() throws Exception{
          INDArray buff = Nd4j.zeros(new int[]{3}, DataType.BYTE);
          buff.putScalar(0, 97); // a
          buff.putScalar(1, 98); // b
          buff.putScalar(2, 99); // c
          ((BaseDataBuffer)buff.data()).syncToPrimary();


          PythonVariables pyInputs = new PythonVariables();
          pyInputs.addBytes("buff", new BytePointer(buff.data().pointer()));

          PythonVariables pyOutputs = new PythonVariables();
          pyOutputs.addBytes("buff"); // same name as input, because inplace update

          String code = "buff[0]=99\nbuff[1]=98\nbuff[2]=97";
          Python.exec(code, pyInputs, pyOutputs);
          Assert.assertEquals("cba", pyOutputs.getBytesValue("buff").getString());
          Assert.assertEquals(buff.data().address(), pyOutputs.getBytesValue("buff").address());
      }

    @Test
    public void testByteBufferInplace() throws Exception{
        INDArray buff = Nd4j.zeros(new int[]{3}, DataType.BYTE);
        buff.putScalar(0, 97); // a
        buff.putScalar(1, 98); // b
        buff.putScalar(2, 99); // c
        ((BaseDataBuffer)buff.data()).syncToPrimary();

        PythonVariables pyInputs = new PythonVariables();
        pyInputs.addBytes("buff", new BytePointer(buff.data().pointer()));
        String code = "buff[0]+=2\nbuff[2]-=2";
        Python.exec(code, pyInputs, null);
        Assert.assertEquals("cba", pyInputs.getBytesValue("buff").getString());
        INDArray expected = buff.dup();
        expected.putScalar(0, 99);
        expected.putScalar(2, 97);
        Assert.assertEquals(buff, expected);

    }

    @Test
    public void testByteBufferOutputWithCopy() throws Exception{
        INDArray buff = Nd4j.zeros(new int[]{3}, DataType.BYTE);
        buff.putScalar(0, 97); // a
        buff.putScalar(1, 98); // b
        buff.putScalar(2, 99); // c
        ((BaseDataBuffer)buff.data()).syncToPrimary();


        PythonVariables pyInputs = new PythonVariables();
        pyInputs.addBytes("buff", new BytePointer(buff.data().pointer()));

        PythonVariables pyOutputs = new PythonVariables();
        pyOutputs.addBytes("out");

        String code = "buff[0]=99\nbuff[1]=98\nbuff[2]=97\nout=bytes(buff)";
        Python.exec(code, pyInputs, pyOutputs);
        Assert.assertEquals("cba", pyOutputs.getBytesValue("out").getString());
    }

    @Test
    public void testDoubleDeviceAllocation() throws Exception{
        if(!"CUDA".equalsIgnoreCase(Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend"))){
            return;
        }
        // Test to make sure that multiple device buffers are not allocated
        // for the same host buffer
        INDArray arr = Nd4j.rand(3, 2);
        ((BaseDataBuffer)arr.data()).syncToPrimary();
        long deviceAddress1 = getDeviceAddress(arr);
        PythonVariables pyInputs = new PythonVariables();
        pyInputs.addNDArray("arr", arr);
        PythonVariables pyOutputs = new PythonVariables();
        pyOutputs.addNDArray("arr");
        String code = "arr += 2";
        Python.exec(code, pyInputs, pyOutputs);
        INDArray arr2 = pyOutputs.getNDArrayValue("arr");
        long deviceAddress2 = getDeviceAddress(arr2);
        Assert.assertEquals(deviceAddress1, deviceAddress2);


    }

    @Test
    public void testBadCode() throws Exception{
        Python.setContext("badcode");
        PythonVariables pyInputs = new PythonVariables();
        PythonVariables pyOutputs = new PythonVariables();

        pyInputs.addNDArray("x", Nd4j.zeros(DataType.LONG, 2, 3));
        pyInputs.addNDArray("y", Nd4j.ones(DataType.LONG, 2, 3));
        pyOutputs.addNDArray("z");

        String code = "z = x + a";

        try{
            Python.exec(code, pyInputs, pyOutputs);
            fail("No exception thrown");
        } catch (PythonException pe ){
            Assert.assertEquals("NameError: name 'a' is not defined", pe.getMessage());
        }

        Python.setMainContext();
    }

    @Test
    public void testIsNone(){
        PythonObject d = Python.dict();
        PythonObject none = d.attr("get").call("x");
        Assert.assertTrue(none.isNone());
        d.set(new PythonObject("x"), new PythonObject("y"));
        PythonObject notNone = d.attr("get").call("x");
        Assert.assertFalse(notNone.isNone());
        Assert.assertEquals("y", notNone.toString());
    }

    private static long getDeviceAddress(INDArray array){
        if(!"CUDA".equalsIgnoreCase(Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend"))){
            throw new IllegalStateException("Cannot ge device pointer for non-CUDA device");
        }

        //Use reflection here as OpaqueDataBuffer is only available on BaseCudaDataBuffer and BaseCpuDataBuffer - not DataBuffer/BaseDataBuffer
        // due to it being defined in nd4j-native-api, not nd4j-api
        try {
            Class<?> c = Class.forName("org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer");
            Method m = c.getMethod("getOpaqueDataBuffer");
            OpaqueDataBuffer db = (OpaqueDataBuffer) m.invoke(array.data());
            long address = db.specialBuffer().address();
            return address;
        } catch (Throwable t){
            throw new RuntimeException("Error getting OpaqueDataBuffer", t);
        }
    }

}
