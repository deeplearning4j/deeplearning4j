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
        TF_Tensor tf_tensor = TensorflowConversion.tensorFromNDArray(arr);
        INDArray fromTensor = TensorflowConversion.ndArrayFromTensor(tf_tensor);
        arr.addi(1.0);
        assertEquals(arr,fromTensor);
    }


}
