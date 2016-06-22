package org.nd4j.linalg.api.ndarray;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.io.Assert;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;

/**
 * Created by susaneraly on 6/18/16.
 */
@RunWith(Parameterized.class)
public class TestJSONC extends BaseNd4jTest{
    public TestJSONC(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void TestReadWrite() {
        INDArray origArr = Nd4j.rand('c',10,10).muli(100); //since we write only two decimal points..
        NdArrayJSONWriter jsonWriter = new NdArrayJSONWriter();
        jsonWriter.write(origArr,"someArr.json");

        NdArrayJSONReader jsonReader = new NdArrayJSONReader();
        INDArray readBack = jsonReader.read(new File("someArr.json"));
        System.out.println("=========================================================================");
        System.out.println(origArr);
        System.out.println("=========================================================================");
        System.out.println(readBack);
        Assert.isTrue(Transforms.abs(origArr.subi(readBack)).maxNumber().doubleValue() < 0.09);
    }
    @Test
    public void TestReadWriteSimple() {
        INDArray origArr = Nd4j.rand(1,1).muli(100); //since we write only two decimal points..
        NdArrayJSONWriter jsonWriter = new NdArrayJSONWriter();
        jsonWriter.write(origArr,"someArr.json");

        NdArrayJSONReader jsonReader = new NdArrayJSONReader();
        INDArray readBack = jsonReader.read(new File("someArr.json"));
        System.out.println("=========================================================================");
        System.out.println(origArr);
        System.out.println("=========================================================================");
        System.out.println(readBack);
        Assert.isTrue(Transforms.abs(origArr.subi(readBack)).maxNumber().doubleValue() < 0.09);
    }
    @Test
    public void TestReadWriteNd() {
        INDArray origArr = Nd4j.rand(13,2,11,3,7,19).muli(100); //since we write only two decimal points..
        NdArrayJSONWriter jsonWriter = new NdArrayJSONWriter();
        jsonWriter.write(origArr,"someArr.json");

        NdArrayJSONReader jsonReader = new NdArrayJSONReader();
        INDArray readBack = jsonReader.read(new File("someArr.json"));
        System.out.println("=========================================================================");
        System.out.println(origArr);
        System.out.println("=========================================================================");
        System.out.println(readBack);
        Assert.isTrue(Transforms.abs(origArr.subi(readBack)).maxNumber().doubleValue() < 0.09);
    }
    @Test
    public void TestWierdShape() {
        INDArray origArr = Nd4j.rand(1,1,2,1,1).muli(100); //since we write only two decimal points..
        NdArrayJSONWriter jsonWriter = new NdArrayJSONWriter();
        jsonWriter.write(origArr,"someArr.json");

        NdArrayJSONReader jsonReader = new NdArrayJSONReader();
        INDArray readBack = jsonReader.read(new File("someArr.json"));
        System.out.println("=========================================================================");
        System.out.println(origArr);
        System.out.println("=========================================================================");
        System.out.println(readBack);
        Assert.isTrue(Transforms.abs(origArr.subi(readBack)).maxNumber().doubleValue() < 0.09);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
