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

import org.eclipse.python4j.*;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;


@javax.annotation.concurrent.NotThreadSafe
@RunWith(Parameterized.class)
public class PythonNumpyJobTest {
    private DataType dataType;

    public PythonNumpyJobTest(DataType dataType){
        this.dataType = dataType;
    }

    @Parameterized.Parameters(name = "{index}: Testing with DataType={0}")
    public static DataType[] params() {
        return new DataType[]{
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
    }

    @Test
    public void testNumpyJobBasic(){
        PythonContextManager.deleteNonMainContexts();
        List<PythonVariable> inputs = new ArrayList<>();
        INDArray x = Nd4j.ones(dataType, 2, 3);
        INDArray y = Nd4j.zeros(dataType, 2, 3);
        INDArray z = (dataType == DataType.BOOL)?x:x.mul(y.add(2));
        z = (dataType == DataType.BFLOAT16)? z.castTo(DataType.FLOAT): z;
        PythonType<INDArray> arrType = PythonTypes.get("numpy.ndarray");
        inputs.add(new PythonVariable<>("x", arrType, x));
        inputs.add(new PythonVariable<>("y", arrType, y));
        List<PythonVariable> outputs = new ArrayList<>();
        PythonVariable<INDArray> output = new PythonVariable<>("z", arrType);
        outputs.add(output);
        String code = (dataType == DataType.BOOL)?"z = x":"z = x * (y + 2)";

        PythonJob job = new PythonJob("job1", code, false);

        job.exec(inputs, outputs);

        INDArray z2 = output.getValue();

        if (dataType == DataType.BFLOAT16){
            z2  = z2.castTo(DataType.FLOAT);
        }

        Assert.assertEquals(z, z2);

    }

    @Test
    public void testNumpyJobReturnAllVariables(){
        PythonContextManager.deleteNonMainContexts();
        List<PythonVariable> inputs = new ArrayList<>();
        INDArray x = Nd4j.ones(dataType, 2, 3);
        INDArray y = Nd4j.zeros(dataType, 2, 3);
        INDArray z = (dataType == DataType.BOOL)?x:x.mul(y.add(2));
        PythonType<INDArray> arrType = PythonTypes.get("numpy.ndarray");
        inputs.add(new PythonVariable<>("x", arrType, x));
        inputs.add(new PythonVariable<>("y", arrType, y));
        String code = (dataType == DataType.BOOL)?"z = x":"z = x * (y + 2)";

        PythonJob job = new PythonJob("job1", code, false);
        List<PythonVariable> outputs = job.execAndReturnAllVariables(inputs);

        INDArray x2 = (INDArray) outputs.get(0).getValue();
        INDArray y2 = (INDArray) outputs.get(1).getValue();
        INDArray z2 = (INDArray) outputs.get(2).getValue();

        if (dataType == DataType.BFLOAT16){
            x = x.castTo(DataType.FLOAT);
            y = y.castTo(DataType.FLOAT);
            z = z.castTo(DataType.FLOAT);
        }
        Assert.assertEquals(x, x2);
        Assert.assertEquals(y, y2);
        Assert.assertEquals(z, z2);

    }


    @Test
    public void testMultipleNumpyJobsParallel(){
        PythonContextManager.deleteNonMainContexts();
        String code1 =(dataType == DataType.BOOL)?"z = x":"z = x + y";
        PythonJob job1 = new PythonJob("job1", code1, false);

        String code2 =(dataType == DataType.BOOL)?"z = y":"z = x - y";
        PythonJob job2 = new PythonJob("job2", code2, false);

        List<PythonVariable> inputs = new ArrayList<>();
        INDArray x = Nd4j.ones(dataType, 2, 3);
        INDArray y = Nd4j.zeros(dataType, 2, 3);
        INDArray z1 = (dataType == DataType.BOOL)?x:x.add(y);
        z1 = (dataType == DataType.BFLOAT16)? z1.castTo(DataType.FLOAT): z1;
        INDArray z2 = (dataType == DataType.BOOL)?y:x.sub(y);
        z2 = (dataType == DataType.BFLOAT16)? z2.castTo(DataType.FLOAT): z2;
        PythonType<INDArray> arrType = PythonTypes.get("numpy.ndarray");
        inputs.add(new PythonVariable<>("x", arrType, x));
        inputs.add(new PythonVariable<>("y", arrType, y));


        List<PythonVariable> outputs = new ArrayList<>();

        outputs.add(new PythonVariable<>("z", arrType));

        job1.exec(inputs, outputs);

        assertEquals(z1, outputs.get(0).getValue());


        job2.exec(inputs, outputs);

        assertEquals(z2, outputs.get(0).getValue());

    }


    @Test
    public synchronized void  testNumpyJobSetupRun(){
        if (dataType == DataType.BOOL)return;
        PythonContextManager.deleteNonMainContexts();
        String code = "five=None\n" +
                "def setup():\n" +
                "    global five\n"+
                "    five = 5\n\n" +
                "def run(a, b):\n" +
                "    c = a + b + five\n"+
                "    return {'c':c}\n\n";

        PythonJob job = new PythonJob("job1", code, true);

        List<PythonVariable> inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("a", NumpyArray.INSTANCE, Nd4j.ones(dataType, 2, 3).mul(2)));
        inputs.add(new PythonVariable<>("b", NumpyArray.INSTANCE, Nd4j.ones(dataType, 2, 3).mul(3)));

        List<PythonVariable> outputs = new ArrayList<>();
        outputs.add(new PythonVariable<>("c", NumpyArray.INSTANCE));
        job.exec(inputs, outputs);


        assertEquals(Nd4j.ones((dataType == DataType.BFLOAT16)? DataType.FLOAT: dataType, 2, 3).mul(10),
                outputs.get(0).getValue());


        inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("a", NumpyArray.INSTANCE, Nd4j.ones(dataType, 2, 3).mul(3)));
        inputs.add(new PythonVariable<>("b", NumpyArray.INSTANCE, Nd4j.ones(dataType, 2, 3).mul(4)));


        outputs = new ArrayList<>();
        outputs.add(new PythonVariable<>("c", NumpyArray.INSTANCE));

        job.exec(inputs, outputs);

        assertEquals(Nd4j.ones((dataType == DataType.BFLOAT16)? DataType.FLOAT: dataType, 2, 3).mul(12),
                outputs.get(0).getValue());


    }
    @Test
    public void testNumpyJobSetupRunAndReturnAllVariables(){
        if (dataType == DataType.BOOL)return;
        PythonContextManager.deleteNonMainContexts();
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
        inputs.add(new PythonVariable<>("a", NumpyArray.INSTANCE, Nd4j.ones(dataType, 2, 3).mul(2)));
        inputs.add(new PythonVariable<>("b", NumpyArray.INSTANCE, Nd4j.ones(dataType, 2, 3).mul(3)));

        List<PythonVariable> outputs = job.execAndReturnAllVariables(inputs);

        assertEquals(Nd4j.ones((dataType == DataType.BFLOAT16)? DataType.FLOAT: dataType, 2, 3).mul(10),
                outputs.get(1).getValue());


        inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("a", NumpyArray.INSTANCE, Nd4j.ones(dataType, 2, 3).mul(3)));
        inputs.add(new PythonVariable<>("b", NumpyArray.INSTANCE, Nd4j.ones(dataType, 2, 3).mul(4)));


        outputs = job.execAndReturnAllVariables(inputs);


        assertEquals(Nd4j.ones((dataType == DataType.BFLOAT16)? DataType.FLOAT: dataType, 2, 3).mul(12),
                outputs.get(1).getValue());


    }

    @Test
    public void testMultipleNumpyJobsSetupRunParallel(){
        if (dataType == DataType.BOOL)return;
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

        List<PythonVariable> inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("a", NumpyArray.INSTANCE, Nd4j.ones(dataType, 2, 3).mul(2)));
        inputs.add(new PythonVariable<>("b", NumpyArray.INSTANCE, Nd4j.ones(dataType, 2, 3).mul(3)));


        List<PythonVariable> outputs = new ArrayList<>();
        outputs.add(new PythonVariable<>("c", NumpyArray.INSTANCE));

        job1.exec(inputs, outputs);

        assertEquals(Nd4j.ones((dataType == DataType.BFLOAT16)? DataType.FLOAT: dataType, 2, 3).mul(10),
                outputs.get(0).getValue());


        job2.exec(inputs, outputs);

        assertEquals(Nd4j.zeros((dataType == DataType.BFLOAT16)? DataType.FLOAT: dataType, 2, 3),
                outputs.get(0).getValue());


        inputs = new ArrayList<>();
        inputs.add(new PythonVariable<>("a", NumpyArray.INSTANCE, Nd4j.ones(dataType, 2, 3).mul(3)));
        inputs.add(new PythonVariable<>("b", NumpyArray.INSTANCE, Nd4j.ones(dataType, 2, 3).mul(4)));

        outputs = new ArrayList<>();
        outputs.add(new PythonVariable<>("c", NumpyArray.INSTANCE));


        job1.exec(inputs, outputs);

        assertEquals(Nd4j.ones((dataType == DataType.BFLOAT16)? DataType.FLOAT: dataType, 2, 3).mul(12),
                outputs.get(0).getValue());


        job2.exec(inputs, outputs);

        assertEquals(Nd4j.ones((dataType == DataType.BFLOAT16)? DataType.FLOAT: dataType, 2, 3).mul(2),
                outputs.get(0).getValue());


    }

}
