package org.nd4j.arrow;

import org.apache.arrow.flatbuf.Tensor;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class ArrowSerdeTest {

    @Test
    public void testBackAndForth() {
        INDArray arr = Nd4j.linspace(1,4,4);
        Tensor tensor = ArrowSerde.toTensor(arr);
        INDArray arr2 = ArrowSerde.fromTensor(tensor);
        assertEquals(arr,arr2);
    }



}
