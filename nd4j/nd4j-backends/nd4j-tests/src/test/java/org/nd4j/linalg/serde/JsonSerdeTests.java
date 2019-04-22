package org.nd4j.linalg.serde;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.serde.RowVectorDeserializer;
import org.nd4j.linalg.lossfunctions.serde.RowVectorSerializer;
import org.nd4j.serde.jackson.shaded.NDArrayTextDeSerializer;
import org.nd4j.serde.jackson.shaded.NDArrayTextSerializer;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import static org.junit.Assert.assertEquals;

public class JsonSerdeTests {


    @Test
    public void testNDArrayTextSerializer() throws Exception {
        for(char order : new char[]{'c', 'f'}) {
            Nd4j.factory().setOrder(order);
            for (DataType globalDT : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                Nd4j.setDefaultDataTypes(globalDT, globalDT);

                Nd4j.getRandom().setSeed(12345);
                INDArray in = Nd4j.rand(DataType.DOUBLE, 3, 4).muli(20).subi(10);

                ObjectMapper om = new ObjectMapper();

                for (DataType dt : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF, DataType.LONG, DataType.INT, DataType.SHORT,
                        DataType.BYTE, DataType.UBYTE, DataType.BOOL, DataType.UTF8}) {

                    INDArray arr;
                    if(dt == DataType.UTF8){
                        arr = Nd4j.create("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l").reshape('c', 3, 4);
                    } else {
                        arr = in.castTo(dt);
                    }

                    TestClass tc = new TestClass(arr);

                    String s = om.writeValueAsString(tc);
//                    System.out.println(dt);
//                    System.out.println(s);
//                    System.out.println("\n\n\n");

                    TestClass deserialized = om.readValue(s, TestClass.class);
                    assertEquals(dt.toString(), tc, deserialized);
                }
            }
        }
    }


    @Test
    public void testBackwardCompatability() throws Exception {
        Nd4j.getNDArrayFactory().setOrder('f');

        for(DataType dt : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            Nd4j.setDefaultDataTypes(dt, dt);
            //NDArrayTextDeserializer will be used in ILossFunction instances that used to use RowVectorSerializer - and it needs to support old format

            INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
            TestClassRow r = new TestClassRow(arr);

            ObjectMapper om = new ObjectMapper();
            String s = om.writeValueAsString(r);

            TestClass tc = om.readValue(s, TestClass.class);

            assertEquals(arr, tc.getArr());

        }

    }


    @EqualsAndHashCode
    @Data
    @NoArgsConstructor
    public static class TestClass {
        @JsonDeserialize(using = NDArrayTextDeSerializer.class)
        @JsonSerialize(using = NDArrayTextSerializer.class)
        public INDArray arr;

        public TestClass(@JsonProperty("arr") INDArray arr){
            this.arr = arr;
        }
    }

    @EqualsAndHashCode
    @Data
    @NoArgsConstructor
    public static class TestClassRow {
        @JsonDeserialize(using = RowVectorDeserializer.class)
        @JsonSerialize(using = RowVectorSerializer.class)
        public INDArray arr;

        public TestClassRow(@JsonProperty("arr") INDArray arr){
            this.arr = arr;
        }
    }

}
