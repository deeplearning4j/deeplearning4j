package org.nd4j.tensorflow.conversion;

import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import static org.bytedeco.javacpp.tensorflow.*;
import static org.bytedeco.javacpp.tensorflow.DT_INT64;
import static org.bytedeco.javacpp.tensorflow.TF_NewTensor;
import static org.junit.Assert.assertEquals;

public class TensorflowConversionTest {


    @Test
    public void testConversionFromNdArray() throws Exception {
        INDArray arr = Nd4j.linspace(1,4,4);
        TF_Tensor tf_tensor = TensorflowConversion.tensorFromNDArray(arr);
        INDArray fromTensor = TensorflowConversion.ndArrayFromTensor(tf_tensor);
        assertEquals(arr,fromTensor);
    }


}
