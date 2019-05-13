package org.nd4j.linalg;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

import static org.junit.Assert.assertEquals;

public class DataTypeTest {

    @Test
    public void testDataTypes() throws IOException {
        for (DataType type : DataType.values()) {
            INDArray in1 = Nd4j.ones(type);

            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos);
            oos.writeObject(in1);

            ByteArrayInputStream bios = new ByteArrayInputStream(baos.toByteArray());
            ObjectInputStream ois = new ObjectInputStream(bios);
            INDArray in2 = null;
            try {
                in2 = (INDArray) ois.readObject();
            }
            catch (ClassNotFoundException e) {

            }

            assertEquals(in1, in2);

        }
    }
}
