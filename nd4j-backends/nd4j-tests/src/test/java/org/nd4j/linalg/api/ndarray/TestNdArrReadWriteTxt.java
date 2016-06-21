
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
public class TestNdArrReadWriteTxt extends BaseNd4jTest{

    public TestNdArrReadWriteTxt(Nd4jBackend backend) {

        super(backend);
    }

    @Test
    public void TestReadWrite() {
        INDArray origArr = Nd4j.rand('f',10,10).muli(100); //since we write only two decimal points..
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

    //Repeating tests with precision and separator
    @Test
    public void TestReadWriteSepPrec() {
        INDArray origArr = Nd4j.rand('f',3,3).muli(1000); //since we write only four decimal points..
        Nd4j.writeTxt(origArr, "someArrNew.txt",":",3);
        INDArray readBack = Nd4j.readTxt("someArrNew.txt",":");
        System.out.println("=========================================================================");
        System.out.println(origArr);
        System.out.println("=========================================================================");
        System.out.println(readBack);
        Assert.isTrue(Transforms.abs(origArr.subi(readBack)).maxNumber().doubleValue() < 0.001);
    }
    @Test
    public void TestReadWriteSimpleSepPrec() {
        INDArray origArr = Nd4j.rand(1,1).muli(1000); //since we write only two decimal points..
        Nd4j.writeTxt(origArr, "someArr.txt","_",3);
        INDArray readBack = Nd4j.readTxt("someArr.txt","_");
        System.out.println("=========================================================================");
        System.out.println(origArr);
        System.out.println("=========================================================================");
        System.out.println(readBack);
        Assert.isTrue(Transforms.abs(origArr.subi(readBack)).maxNumber().doubleValue() < 0.001);
    }
    @Test
    public void TestReadWriteNdSepPrec() {
        INDArray origArr = Nd4j.rand(13,2,11,3,7,19).muli(10000); //since we write only two decimal points..
        Nd4j.writeTxt(origArr, "someArr.txt","_",4);
        INDArray readBack = Nd4j.readTxt("someArr.txt","_");
        System.out.println("=========================================================================");
        System.out.println(origArr);
        System.out.println("=========================================================================");
        System.out.println(readBack);
        Assert.isTrue(Transforms.abs(origArr.subi(readBack)).maxNumber().doubleValue() < 0.0001);
    }
    @Test
    public void TestWierdShapeSepPrec() {
        INDArray origArr = Nd4j.rand(1,1,2,1,1).muli(10000); //since we write only two decimal points..
        Nd4j.writeTxt(origArr, "someArr.txt","_",4);
        INDArray readBack = Nd4j.readTxt("someArr.txt","_");
        System.out.println("=========================================================================");
        System.out.println(origArr);
        System.out.println("=========================================================================");
        System.out.println(readBack);
        Assert.isTrue(Transforms.abs(origArr.subi(readBack)).maxNumber().doubleValue() < 0.0001);
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
