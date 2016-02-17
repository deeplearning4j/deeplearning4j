package org.nd4j.linalg.api.indexing.shape;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.Indices;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.Assert.assertArrayEquals;

/**
 * @author Adam Gibson
 */
public class IndexShapeTests2d extends BaseNd4jTest {
    public IndexShapeTests2d() {
    }

    public IndexShapeTests2d(Nd4jBackend backend) {
        super(backend);
    }

    public IndexShapeTests2d(String name) {
        super(name);
    }

    public IndexShapeTests2d(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    private int[] shape = {3,2};


    @Test
    public void test2dCases() {
        assertArrayEquals(new int[]{1,2},Indices.shape(shape,new INDArrayIndex[]{NDArrayIndex.point(1)}));
        assertArrayEquals(new int[]{3,1},Indices.shape(shape,new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(1)}));
    }

    @Test
    public void testNewAxis2d() {
        assertArrayEquals(new int[]{1,3,2},Indices.shape(shape,new INDArrayIndex[]{NDArrayIndex.newAxis(),NDArrayIndex.all(),NDArrayIndex.all()}));
        assertArrayEquals(new int[]{3,1,2},Indices.shape(shape,new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.newAxis(),NDArrayIndex.all()}));

    }


    @Override
    public char ordering() {
        return 'f';
    }
}
