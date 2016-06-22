package org.nd4j.linalg.api.ndarray;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.io.Assert;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Created by susaneraly on 6/18/16.
 */
@RunWith(Parameterized.class)
public class TestNdArrReadWriteTxtC extends BaseNd4jTest{
    public TestNdArrReadWriteTxtC(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void TestReadWrite() {
        INDArray origArr = Nd4j.rand('c',10,10).muli(100); //since we write only two decimal points..
        Nd4j.writeTxt(origArr, "someArr.txt");
        INDArray readBack = Nd4j.readTxt("someArr.txt");
        System.out.println("=========================================================================");
        System.out.println(origArr);
        System.out.println("=========================================================================");
        System.out.println(readBack);
        Assert.isTrue(Transforms.abs(origArr.subi(readBack)).maxNumber().doubleValue() < 0.01);
    }
    @Test
    public void TestReadWriteSimple() {
        INDArray origArr = Nd4j.rand(1,1).muli(100); //since we write only two decimal points..
        Nd4j.writeTxt(origArr, "someArr.txt");
        INDArray readBack = Nd4j.readTxt("someArr.txt");
        System.out.println("=========================================================================");
        System.out.println(origArr);
        System.out.println("=========================================================================");
        System.out.println(readBack);
        Assert.isTrue(Transforms.abs(origArr.subi(readBack)).maxNumber().doubleValue() < 0.01);
    }
    @Test
    public void TestReadWriteNd() {
        INDArray origArr = Nd4j.rand(13,2,11,3,7,19).muli(100); //since we write only two decimal points..
        Nd4j.writeTxt(origArr, "someArr.txt");
        INDArray readBack = Nd4j.readTxt("someArr.txt");
        System.out.println("=========================================================================");
        System.out.println(origArr);
        System.out.println("=========================================================================");
        System.out.println(readBack);
        Assert.isTrue(Transforms.abs(origArr.subi(readBack)).maxNumber().doubleValue() < 0.01);
    }
    @Test
    public void TestWierdShape() {
        INDArray origArr = Nd4j.rand(1,1,2,1,1).muli(100); //since we write only two decimal points..
        Nd4j.writeTxt(origArr, "someArr.txt");
        INDArray readBack = Nd4j.readTxt("someArr.txt");
        System.out.println("=========================================================================");
        System.out.println(origArr);
        System.out.println("=========================================================================");
        System.out.println(readBack);
        Assert.isTrue(Transforms.abs(origArr.subi(readBack)).maxNumber().doubleValue() < 0.01);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
