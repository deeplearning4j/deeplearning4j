package org.nd4j.linalg.serde;

import lombok.AllArgsConstructor;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.jackson.shaded.NDArrayTextDeSerializer;
import org.nd4j.serde.jackson.shaded.NDArrayTextSerializer;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import static org.junit.Assert.assertEquals;

public class JsonSerdeTests {


    @Test
    public void testNDArrayTextSerializer() throws Exception {

        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.rand(DataType.DOUBLE, 3, 4).muli(20).subi(10);

        ObjectMapper om = new ObjectMapper();

        for(DataType dt : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF, DataType.LONG, DataType.INT, DataType.SHORT,
                DataType.BYTE, DataType.UBYTE, DataType.BOOL}){

            INDArray arr = in.castTo(dt);

            TestClass tc = new TestClass(arr);

            String s = om.writeValueAsString(tc);
            System.out.println(dt);
            System.out.println(s);
            System.out.println("\n\n\n");

            INDArray deserialized = om.readValue(s, INDArray.class);
            assertEquals(dt.toString(), arr, deserialized);
        }

    }

    @AllArgsConstructor
    public static class TestClass {

        @JsonDeserialize(using = NDArrayTextDeSerializer.class)
        @JsonSerialize(using = NDArrayTextSerializer.class)
        public INDArray arr;

    }

}
