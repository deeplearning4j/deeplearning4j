package org.nd4j.linalg.compression;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

@Slf4j
@RunWith(Parameterized.class)
public class ConversionTests extends BaseNd4jTest {

    public ConversionTests(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testDoubleToFloats1() {
        val dtype = Nd4j.dataType();

        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        val arrayX = Nd4j.create(10).assign(1.0);


        Nd4j.setDataType(DataBuffer.Type.FLOAT);
        val arrayY = Nd4j.create(10).assign(1.0);


        val converted = arrayX.convertToFloats();
        val exp = Nd4j.create(10).assign(2.0);
        converted.addi(arrayY);

        assertEquals(exp, converted);



        Nd4j.setDataType(dtype);
    }


    @Test
    public void testFloatsToDoubles1() {
        val dtype = Nd4j.dataType();

        Nd4j.setDataType(DataBuffer.Type.FLOAT);
        val arrayX = Nd4j.create(10).assign(1.0);


        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        val arrayY = Nd4j.create(10).assign(1.0);


        val converted = arrayX.convertToDoubles();
        val exp = Nd4j.create(10).assign(2.0);
        converted.addi(arrayY);

        assertEquals(exp, converted);



        Nd4j.setDataType(dtype);
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
