package org.nd4j.linalg.api;

import org.nd4j.linalg.primitives.Pair;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.util.ArrayUtil;

import static org.junit.Assert.assertArrayEquals;

/**
 * Created by Alex on 30/04/2016.
 */
public class TestNDArrayCreationUtil extends BaseNd4jTest {


    public TestNDArrayCreationUtil(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testShapes() {

        // FIXME: int cast
        long[] shape2d = {2, 3};
        for (Pair<INDArray, String> p : NDArrayCreationUtil.getAllTestMatricesWithShape(2, 3, 12345)) {
            assertArrayEquals(p.getSecond(), shape2d, p.getFirst().shape());
        }

        // FIXME: int cast
        long[] shape3d = {2, 3, 4};
        for (Pair<INDArray, String> p : NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, shape3d)) {
            assertArrayEquals(p.getSecond(), shape3d, p.getFirst().shape());
        }

        // FIXME: int cast
        long[] shape4d = {2, 3, 4, 5};
        for (Pair<INDArray, String> p : NDArrayCreationUtil.getAll4dTestArraysWithShape(12345, ArrayUtil.toInts(shape4d))) {
            assertArrayEquals(p.getSecond(), shape4d, p.getFirst().shape());
        }

        // FIXME: int cast
        long[] shape5d = {2, 3, 4, 5, 6};
        for (Pair<INDArray, String> p : NDArrayCreationUtil.getAll5dTestArraysWithShape(12345, ArrayUtil.toInts(shape5d))) {
            assertArrayEquals(p.getSecond(), shape5d, p.getFirst().shape());
        }

        // FIXME: int cast
        long[] shape6d = {2, 3, 4, 5, 6, 7};
        for (Pair<INDArray, String> p : NDArrayCreationUtil.getAll6dTestArraysWithShape(12345, ArrayUtil.toInts(shape6d))) {
            assertArrayEquals(p.getSecond(), shape6d, p.getFirst().shape());
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
