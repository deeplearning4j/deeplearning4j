package org.nd4j.linalg.api.indexing;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.ShapeOffsetResolution;
import org.nd4j.linalg.util.ArrayUtil;

import static org.junit.Assert.*;
import static org.junit.Assume.*;


/**
 * @author Adam Gibson
 */
public class ShapeResolutionTestsC extends BaseNd4jTest {
    @Test
    public void testRowVectorShapeOneZeroOffset() {
        INDArray arr = Nd4j.create(2, 2);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(arr);
        //row 0
        resolution.exec(new NDArrayIndex(0));
        int[] oneIndexShape = ArrayUtil.copy(resolution.getShapes());
        assertArrayEquals(new int[]{1,2},oneIndexShape);
        int[] oneIndexOffsets = ArrayUtil.copy(resolution.getOffsets());
        assertArrayEquals(new int[]{0, 0}, oneIndexOffsets);
        assertEquals(0,resolution.getOffset());
        int[] oneIndexStrides = ArrayUtil.copy(resolution.getStrides());
        assertArrayEquals(new int[]{2,1},oneIndexStrides);



    }

    @Test
    public void testRowVectorShapeTwoZeroOffset() {
        INDArray arr = Nd4j.create(2, 2);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(arr);
        //row 0
        resolution.exec(new NDArrayIndex(0),NDArrayIndex.all());
        int[] oneIndexShape = ArrayUtil.copy(resolution.getShapes());
        assertArrayEquals(new int[]{1,2},oneIndexShape);
        int[] oneIndexOffsets = ArrayUtil.copy(resolution.getOffsets());
        assertArrayEquals(new int[]{0, 0}, oneIndexOffsets);
        assertEquals(0,resolution.getOffset());
        int[] oneIndexStrides = ArrayUtil.copy(resolution.getStrides());
        assertArrayEquals(new int[]{2,1},oneIndexStrides);

    }



    @Test
    public void testRowVectorShapeOneOneOffset() {
        INDArray arr = Nd4j.create(2, 2);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(arr);
        //row 0
        resolution.exec(new NDArrayIndex(1));
        int[] oneIndexShape = ArrayUtil.copy(resolution.getShapes());
        assertArrayEquals(new int[]{1,2},oneIndexShape);
        int[] oneIndexOffsets = ArrayUtil.copy(resolution.getOffsets());
        assertArrayEquals(new int[]{1,0}, oneIndexOffsets);
        assertEquals(2,resolution.getOffset());
        int[] oneIndexStrides = ArrayUtil.copy(resolution.getStrides());
        assertArrayEquals(new int[]{2,1},oneIndexStrides);

    }



    @Test
    public void testRowVectorShapeTwoOneOffset() {
        INDArray arr = Nd4j.create(2, 2);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(arr);
        //row 0
        resolution.exec(new NDArrayIndex(1), NDArrayIndex.all());
        int[] oneIndexShape = ArrayUtil.copy(resolution.getShapes());
        assertArrayEquals(new int[]{1,2},oneIndexShape);
        int[] oneIndexOffsets = ArrayUtil.copy(resolution.getOffsets());
        assertArrayEquals(new int[]{1,0}, oneIndexOffsets);
        assertEquals(2,resolution.getOffset());
        int[] oneIndexStrides = ArrayUtil.copy(resolution.getStrides());
        assertArrayEquals(new int[]{2,1},oneIndexStrides);

    }


    @Test
    public void testColumnVectorShapeZeroOffset() {
        INDArray arr = Nd4j.create(2, 2);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(arr);
        resolution.exec(NDArrayIndex.all(), new NDArrayIndex(0));
        assertEquals(0, resolution.getOffset());
        int[] strides = resolution.getStrides();
        assertArrayEquals(new int[]{2,1},resolution.getShapes());
        assertArrayEquals(new int[]{2,1},strides);
    }

    @Test
    public void testColumnVectorShapeOneOffset() {
        INDArray arr = Nd4j.linspace(1,4, 4).reshape(2,2);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(arr);
        resolution.exec(NDArrayIndex.all(), new NDArrayIndex(1));
        assertEquals(1, resolution.getOffset());
        int[] strides = resolution.getStrides();
        assertArrayEquals(new int[]{2,1},resolution.getShapes());
        assertArrayEquals(new int[]{2,1},strides);
    }



    @Override
    public char ordering() {
        return 'c';
    }
}
