package org.nd4j.serde.gson;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class GsonDeserializationUtilsTest {
    @Test
    public void deserializeRawJson_PassInIndArray_ExpectCorrectDeserialization() {
        String serializedRawArray =
                "[[[1.00, 11.00, 3.00],\n" +
                "[13.00, 5.00, 15.00],\n" +
                "[7.00, 17.00, 9.00]]]";
        INDArray expectedArray = buildExpectedArray();

        INDArray indArray = GsonDeserializationUtils.deserializeRawJson(serializedRawArray);

        assertEquals(expectedArray, indArray);
    }

    @Test
    public void deserializeRawJson_ArrayHasOnlyOneRowWithColumns_ExpectCorrectDeserialization() {
        String serializedRawArray = "[1.00, 11.00, 3.00]";
        INDArray expectedArray = Nd4j.create(new double[] { 1, 11, 3 });

        INDArray indArray = GsonDeserializationUtils.deserializeRawJson(serializedRawArray);

        assertEquals(expectedArray, indArray);
    }

    private INDArray buildExpectedArray() {
        INDArray expectedArray = Nd4j.create(3, 3);
        expectedArray.putRow(0, Nd4j.create(new double[] { 1, 11, 3 }));
        expectedArray.putRow(1, Nd4j.create(new double[] { 13, 5, 15 }));
        expectedArray.putRow(2, Nd4j.create(new double[] { 7, 17, 9 }));

        return expectedArray.reshape(1, 3, 3);
    }
}
