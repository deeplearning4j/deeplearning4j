package org.nd4j.linalg.shape.indexing;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class IndexingTestsC extends BaseNd4jTest {

    public IndexingTestsC(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testExecSubArray() {
        INDArray nd = Nd4j.create(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3});

        INDArray sub = nd.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2));
        Nd4j.getExecutioner().exec(new ScalarAdd(sub, 2));
        assertEquals(getFailureMessage(), Nd4j.create(new double[][]{
                {3, 4}, {6, 7}
        }), sub);

    }


    @Test
    public void testLinearViewElementWiseMatching() {
        INDArray linspace = Nd4j.linspace(1,4,4).reshape(2, 2);
        INDArray dup = linspace.dup();
        linspace.addi(dup);
    }


    @Test
    public void testGetRows() {
        INDArray arr = Nd4j.linspace(1,9,9).reshape(3,3);
        INDArray testAssertion = Nd4j.create(new double[][]{
                {4, 5},
                {7, 8}
        });

        INDArray test = arr.get(new SpecifiedIndex(1, 2), new SpecifiedIndex(0, 1));
        assertEquals(testAssertion, test);

    }

    @Test
    public void testFirstColumn() {
        INDArray arr = Nd4j.create(new double[][]{
                {5, 7},
                {6, 8}
        });

        INDArray assertion = Nd4j.create(new double[]{5,6});
        INDArray test = arr.get(NDArrayIndex.all(), NDArrayIndex.point(0));
        assertEquals(assertion,test);
    }

    @Test
    public void testMultiRow() {
        INDArray matrix = Nd4j.linspace(1,9,9).reshape(3, 3);
        INDArray assertion = Nd4j.create(new double[][]{
                {4, 7}
        });

        INDArray test = matrix.get(new SpecifiedIndex(1,2),NDArrayIndex.interval(0, 1));
        assertEquals(assertion,test);
    }

    @Test
    public void testPointIndexes() {
        INDArray arr = Nd4j.create(4, 3, 2);
        INDArray get = arr.get(NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.all());
        assertArrayEquals(new int[]{4,2},get.shape());
        INDArray linspaced = Nd4j.linspace(1,24,24).reshape(4,3,2);
        INDArray assertion = Nd4j.create(new double[][]{
                {3, 4},
                {9, 10},
                {15, 16},
                {21, 22}
        });

        INDArray linspacedGet = linspaced.get(NDArrayIndex.all(),NDArrayIndex.point(1),NDArrayIndex.all());
        for(int i = 0; i < linspacedGet.slices(); i++) {
            INDArray sliceI = linspacedGet.slice(i);
            assertEquals(assertion.slice(i),sliceI);
        }
        assertArrayEquals(new int[]{6,1},linspacedGet.stride());
        assertEquals(assertion,linspacedGet);
    }

    @Test
    public void testGetWithVariedStride() {
        int ph = 0;
        int pw = 0;
        int sy = 2;
        int sx = 2;
        int iLim = 8;
        int jLim  = 8;
        int i = 0;
        int j = 0;
        INDArray img = Nd4j.create(new double[]{
                1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
                3, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4
        }, new int[]{1, 1, 8, 8});


        INDArray padded = Nd4j.pad(img, new int[][]{
                {0, 0}
                , {0, 0}
                , {ph, ph + sy - 1}
                , {pw, pw + sx - 1}}
                , Nd4j.PadMode.CONSTANT);

        INDArray get = padded.get(
                NDArrayIndex.all()
                , NDArrayIndex.all()
                , NDArrayIndex.interval(i, sy, iLim)
                , NDArrayIndex.interval(j, sx, jLim));
        assertArrayEquals(new int[]{81, 81, 18, 2}, get.stride());
        INDArray assertion = Nd4j.create(new double[]{1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3}, new int[]{1, 1, 4, 4});
        assertEquals(assertion, get);

        i = 1;
        iLim = 9;
        INDArray get3  = padded.get(
                NDArrayIndex.all()
                , NDArrayIndex.all()
                , NDArrayIndex.interval(i, sy, iLim)
                , NDArrayIndex.interval(j, sx, jLim));

        INDArray assertion2 = Nd4j.create(new double[]{2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4}, new int[]{1, 1, 4, 4});
        assertArrayEquals(new int[]{81, 81, 18, 2}, get3.stride());
        assertEquals(assertion2, get3);



        i = 0;
        iLim = 8;
        jLim = 9;
        j = 1;
        INDArray get2 = padded.get(
                NDArrayIndex.all()
                , NDArrayIndex.all()
                , NDArrayIndex.interval(i, sy, iLim)
                , NDArrayIndex.interval(j, sx, jLim));
        assertArrayEquals(new int[]{81, 81, 18, 2}, get2.stride());
        assertEquals(assertion, get2);



    }


    @Test
    public void testRowVectorInterval(){
        int len = 30;
        INDArray row = Nd4j.zeros(len);
        for( int i=0; i<len; i++ ){
            row.putScalar(i,i);
        }

        INDArray first10a = row.get(NDArrayIndex.point(0),NDArrayIndex.interval(0, 10));
        assertArrayEquals(first10a.shape(),new int[]{1,10});
        for( int i = 0; i < 10; i++ )
            assertTrue(first10a.getDouble(i) == i);

        INDArray first10b = row.get(NDArrayIndex.interval(0,10));
        assertArrayEquals(first10b.shape(),new int[]{1,10});
        for( int i = 0; i < 10; i++ )
            assertTrue(first10b.getDouble(i) == i);

        INDArray last10a = row.get(NDArrayIndex.point(0),NDArrayIndex.interval(20,30));
        assertArrayEquals(last10a.shape(),new int[]{1,10});
        for( int i = 0; i < 10; i++ )
            assertTrue(last10a.getDouble(i) == 20+i);

        INDArray last10b = row.get(NDArrayIndex.interval(20, 30));
        assertArrayEquals(last10b.shape(),new int[]{1,10});
        for( int i = 0; i < 10; i++ )
            assertTrue(last10b.getDouble(i) == 20+i);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
