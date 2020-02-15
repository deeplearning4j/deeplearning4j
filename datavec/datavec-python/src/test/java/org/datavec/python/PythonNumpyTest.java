package org.datavec.python;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static junit.framework.TestCase.assertEquals;

@RunWith(Parameterized.class)
public class PythonNumpyTest {

    @Parameterized.Parameters(name = "{index}: Testing with DataType={0}")
    public static DataType[] data() {
        return new DataType[] {
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

    private DataType dataType;

    public PythonNumpyTest(DataType dataType) {
        this.dataType = dataType;
    }

    @Test
    public void numpyAndNd4jConversions() throws Exception {
        INDArray input = Nd4j.ones(dataType, 2, 2, 2);

        PythonVariables inputs = new PythonVariables();
        inputs.addNDArray("x", input);

        PythonVariables outputs = new PythonVariables();
        outputs.addNDArray("y");

        PythonJob pythonJob = new PythonJob(String.format("job_%s", dataType.name()) + dataType.name(), "y = x", false);

        pythonJob.exec(inputs, outputs);

        INDArray output = outputs.getNDArrayValue("y");
        // As numpy doesn't support BFLOAT16 we'll convert it to FLOAT
        assertEquals(dataType == DataType.BFLOAT16 ? Nd4j.create(input.data().asFloat(), input.shape(), DataType.FLOAT) : input,
                output);
    }
}
