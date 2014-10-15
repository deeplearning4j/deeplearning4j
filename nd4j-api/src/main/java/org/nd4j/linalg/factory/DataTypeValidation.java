package org.nd4j.linalg.factory;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Data type validation
 * @author Adam Gibson
 */
public class DataTypeValidation {
    public static void assertDouble(INDArray...d) {
        for(INDArray d1 : d)
            assertDouble(d1);
    }
    public static void assertFloat(INDArray...d2) {
        for(INDArray d3 : d2)
            assertFloat(d3);
    }

    public static void assertDouble(INDArray d) {
        assert d.data().dataType().equals(DataBuffer.DOUBLE);
    }
    public static void assertFloat(INDArray d2) {
        assert d2.data().dataType().equals(DataBuffer.FLOAT);
    }

    public static void assertSameDataType(INDArray...indArrays) {
        if(indArrays == null || indArrays.length < 2)
            return;
        String type = indArrays[0].data().dataType();
        for(int i = 1; i < indArrays.length; i++) {
            assert indArrays[i].data().dataType().equals(type);
        }
    }



}
