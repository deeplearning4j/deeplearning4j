package org.nd4j.tensorflow.conversion;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.bytedeco.javacpp.tensorflow.TF_Tensor;
import static org.junit.Assert.assertEquals;

public class TensorflowConversionTest {


    @Test
    public void testConversionFromNdArray() throws Exception {
        INDArray arr = Nd4j.linspace(1,4,4);
        TensorflowConversion tensorflowConversion = new TensorflowConversion();
        TF_Tensor tf_tensor = tensorflowConversion.tensorFromNDArray(arr);
        INDArray fromTensor = tensorflowConversion.ndArrayFromTensor(tf_tensor);
        assertEquals(arr,fromTensor);
        Thread.sleep(5000);
    }


}
