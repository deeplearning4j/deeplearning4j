/*
 *  ******************************************************************************
 *  *
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
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import javax.annotation.concurrent.NotThreadSafe;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

@NotThreadSafe
@RunWith(Parameterized.class)
public class PythonNumpyBasicTest {
    private DataType dataType;
    private long[] shape;

    public PythonNumpyBasicTest(DataType dataType, long[] shape, String dummyArg) {
        this.dataType = dataType;
        this.shape = shape;
    }

    @Parameterized.Parameters(name = "{index}: Testing with DataType={0}, shape={2}")
    public static Collection params() {
        DataType[] types = new DataType[] {
                DataType.BOOL,
                DataType.FLOAT16,
                DataType.BFLOAT16,
                DataType.FLOAT,
                DataType.DOUBLE,
                DataType.INT8,
                DataType.INT16,
                DataType.INT32,
                DataType.INT64,
                DataType.UINT8,
                DataType.UINT16,
                DataType.UINT32,
                DataType.UINT64
        };

        long[][] shapes = new long[][]{
             new long[]{2, 3},
             new long[]{3},
             new long[]{1},
                new long[]{} // scalar
        };


        List<Object[]> ret = new ArrayList<>();
        for (DataType type: types){
            for (long[] shape: shapes){
                ret.add(new Object[]{type, shape, Arrays.toString(shape)});
            }
        }
        return ret;
    }

    @Test
    public void testConversion(){
      try(PythonGIL pythonGIL = PythonGIL.lock()) {
          INDArray arr = Nd4j.zeros(dataType, shape);
          PythonObject npArr = PythonTypes.convert(arr);
          INDArray arr2 = PythonTypes.<INDArray>getPythonTypeForPythonObject(npArr).toJava(npArr);
          if (dataType == DataType.BFLOAT16){
              arr = arr.castTo(DataType.FLOAT);
          }
          Assert.assertEquals(arr,arr2);
      }

    }


    @Test
    public void testExecution() {
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            List<PythonVariable> inputs = new ArrayList<>();
            INDArray x = Nd4j.ones(dataType, shape);
            INDArray y = Nd4j.zeros(dataType, shape);
            INDArray z = (dataType == DataType.BOOL)?x:x.mul(y.add(2));
            z = (dataType == DataType.BFLOAT16)? z.castTo(DataType.FLOAT): z;
            PythonType<INDArray> arrType = PythonTypes.get("numpy.ndarray");
            inputs.add(new PythonVariable<>("x", arrType, x));
            inputs.add(new PythonVariable<>("y", arrType, y));
            List<PythonVariable> outputs = new ArrayList<>();
            PythonVariable<INDArray> output = new PythonVariable<>("z", arrType);
            outputs.add(output);
            String code = (dataType == DataType.BOOL)?"z = x":"z = x * (y + 2)";
            if (shape.length == 0){ // scalar special case
                code += "\nimport numpy as np\nz = np.asarray(float(z), dtype=x.dtype)";
            }
            PythonExecutioner.exec(code, inputs, outputs);
            INDArray z2 = output.getValue();

            Assert.assertEquals(z.dataType(), z2.dataType());
            Assert.assertEquals(z, z2);
        }


    }


    @Test
    public void testInplaceExecution() {
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            if (dataType == DataType.BOOL || dataType == DataType.BFLOAT16)return;
            if (shape.length == 0) return;
            List<PythonVariable> inputs = new ArrayList<>();
            INDArray x = Nd4j.ones(dataType, shape);
            INDArray y = Nd4j.zeros(dataType, shape);
            INDArray z = x.mul(y.add(2));
            // Nd4j.getAffinityManager().ensureLocation(z, AffinityManager.Location.HOST);
            PythonType<INDArray> arrType = PythonTypes.get("numpy.ndarray");
            inputs.add(new PythonVariable<>("x", arrType, x));
            inputs.add(new PythonVariable<>("y", arrType, y));
            List<PythonVariable> outputs = new ArrayList<>();
            PythonVariable<INDArray> output = new PythonVariable<>("x", arrType);
            outputs.add(output);
            String code = "x *= y + 2";
            PythonExecutioner.exec(code, inputs, outputs);
            INDArray z2 = output.getValue();
            Assert.assertEquals(x.dataType(), z2.dataType());
            Assert.assertEquals(z.dataType(), z2.dataType());
            Assert.assertEquals(x, z2);
            Assert.assertEquals(z, z2);
            Assert.assertEquals(x.data().pointer().address(), z2.data().pointer().address());
            if("CUDA".equalsIgnoreCase(Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend"))){
                Assert.assertEquals(getDeviceAddress(x), getDeviceAddress(z2));
            }

        }


    }


    private static long getDeviceAddress(INDArray array) {
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
