package org.nd4j.linalg.api;

import org.apache.commons.math3.util.Pair;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertArrayEquals;

/**
 * Created by Alex on 30/04/2016.
 */
public class TestNDArrayCreationUtil extends BaseNd4jTest {


    public TestNDArrayCreationUtil(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testShapes(){

        int[] shape2d = {2,3};
        for(Pair<INDArray,String> p : NDArrayCreationUtil.getAllTestMatricesWithShape(2,3,12345)){
            assertArrayEquals(p.getSecond(), shape2d, p.getFirst().shape());
        }

        int[] shape3d = {2,3,4};
        for(Pair<INDArray,String> p : NDArrayCreationUtil.getAll3dTestArraysWithShape(12345,shape3d)){
            assertArrayEquals(p.getSecond(), shape3d, p.getFirst().shape());
        }

        int[] shape4d = {2,3,4,5};
        for(Pair<INDArray,String> p : NDArrayCreationUtil.getAll4dTestArraysWithShape(12345,shape4d)){
            assertArrayEquals(p.getSecond(), shape4d, p.getFirst().shape());
        }

        int[] shape5d = {2,3,4,5,6};
        for(Pair<INDArray,String> p : NDArrayCreationUtil.getAll5dTestArraysWithShape(12345,shape5d)){
            assertArrayEquals(p.getSecond(), shape5d, p.getFirst().shape());
        }

        int[] shape6d = {2,3,4,5,6,7};
        for(Pair<INDArray,String> p : NDArrayCreationUtil.getAll6dTestArraysWithShape(12345,shape6d)){
            assertArrayEquals(p.getSecond(), shape6d, p.getFirst().shape());
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
