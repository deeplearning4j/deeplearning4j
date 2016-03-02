package org.nd4j.linalg.api.indexing;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.ShapeOffsetResolution;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

import static org.junit.Assert.*;
import static org.junit.Assert.assertTrue;


/**
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class ShapeResolutionTestsC extends BaseNd4jTest {


    public ShapeResolutionTestsC(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testRowVectorShapeOneZeroOffset() {
        INDArray arr = Nd4j.create(2, 2);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(arr);
        //row 0
        resolution.exec(NDArrayIndex.point(0));
        int[] oneIndexShape = ArrayUtil.copy(resolution.getShapes());
        assertArrayEquals(new int[]{1,2},oneIndexShape);
        int[] oneIndexOffsets = ArrayUtil.copy(resolution.getOffsets());
        assertArrayEquals(new int[]{0, 0}, oneIndexOffsets);
        assertEquals(0,resolution.getOffset());
        int[] oneIndexStrides = ArrayUtil.copy(resolution.getStrides());
        assertArrayEquals(new int[]{1, 1}, oneIndexStrides);

    }

    @Test
    public void testIntervalFirstShapeResolution() {
        INDArray arr = Nd4j.linspace(1,6,6).reshape(3, 2);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(arr);
        resolution.exec(NDArrayIndex.interval(1,3));
        assertFalse(Arrays.equals(arr.shape(), resolution.getShapes()));
    }


    @Test
    public void testRowVectorShapeOneOneOffset() {
        INDArray arr = Nd4j.create(2, 2);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(arr);
        //row 0
        resolution.exec(NDArrayIndex.point(1));
        int[] oneIndexShape = ArrayUtil.copy(resolution.getShapes());
        assertArrayEquals(new int[]{1,2},oneIndexShape);
        assertEquals(2, resolution.getOffset());
        int[] oneIndexStrides = ArrayUtil.copy(resolution.getStrides());
        assertArrayEquals(new int[]{1, 1}, oneIndexStrides);

    }



    @Test
    public void testRowVectorShapeTwoOneOffset() {
        INDArray arr = Nd4j.create(2, 2);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(arr);
        //row 0
        resolution.exec(NDArrayIndex.point(1), NDArrayIndex.all());
        int[] oneIndexShape = ArrayUtil.copy(resolution.getShapes());
        assertArrayEquals(new int[]{1, 2}, oneIndexShape);
        int[] oneIndexOffsets = ArrayUtil.copy(resolution.getOffsets());
        assertArrayEquals(new int[]{0,0}, oneIndexOffsets);
        assertEquals(2,resolution.getOffset());
        int[] oneIndexStrides = ArrayUtil.copy(resolution.getStrides());
        assertArrayEquals(new int[]{1,1},oneIndexStrides);

    }


    @Test
    public void testColumnVectorShapeZeroOffset() {
        INDArray arr = Nd4j.create(2, 2);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(arr);
        resolution.exec(NDArrayIndex.all(), NDArrayIndex.point(0));
        assertEquals(0, resolution.getOffset());
        int[] strides = resolution.getStrides();
        assertArrayEquals(new int[]{2,1},resolution.getShapes());
        assertArrayEquals(new int[]{2,1},strides);
    }

    @Test
    public void testColumnVectorShapeOneOffset() {
        INDArray arr = Nd4j.linspace(1,4, 4).reshape(2, 2);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(arr);
        resolution.exec(NDArrayIndex.all(),  NDArrayIndex.point(1));
        assertEquals(1, resolution.getOffset());
        int[] strides = resolution.getStrides();
        assertArrayEquals(new int[]{2,1},resolution.getShapes());
        assertArrayEquals(new int[]{2,1},strides);
    }


    @Test
    public void testPartiallyOutOfRangeIndices() {
        INDArray arr = Nd4j.linspace(1,4, 4).reshape(2,2);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(arr);
        resolution.exec(NDArrayIndex.interval(0, 2), NDArrayIndex.interval(1, 4));
        assertArrayEquals(new int[]{2,1},resolution.getShapes());
    }

    @Test
    public void testOutOfRangeIndices() {
        INDArray arr = Nd4j.linspace(1,4, 4).reshape(2,2);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(arr);
        try {
            resolution.exec(NDArrayIndex.interval(0, 2), NDArrayIndex.interval(2, 4));
            fail("Out of range index should throw an IllegalArgumentException");
        }catch (IllegalArgumentException e){
            //do nothing
        }
    }

    @Test
    public void testIndexAll(){
        INDArray arr = Nd4j.create(2, 2);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(arr);
        resolution.exec(NDArrayIndex.all(),NDArrayIndex.all());
        assertEquals(resolution.getShapes()[0],2);
        assertEquals(resolution.getShapes()[1],2);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
