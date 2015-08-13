package org.nd4j.linalg.api.indexing;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.Indices;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class IndicesTest2D extends BaseNd4jTest {
    private  int[] shape = {1,1,2,1,3,4,5,1};
    private int[] twoD = {3,2};

    @Test
    public void testSinglePoint() {
        /*
        Assumes all indexes are filled out.
        Test simple general point case
         */
        int[] assertion = {2, 1, 4, 5, 1};
        INDArrayIndex[] indexes = new INDArrayIndex[] {
                NDArrayIndex.point(0),
                NDArrayIndex.point(0),
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.point(0),
                NDArrayIndex.all()
        };

        int[] testShape = Indices.shape(shape,indexes);
        assertArrayEquals(assertion,testShape);
    }

    @Test
    public void testSinglePoin2d() {
        INDArrayIndex[] index = new INDArrayIndex[] {
                NDArrayIndex.point(0),
                NDArrayIndex.all()
        };
        assertArrayEquals(new int[]{1,2},Indices.shape(twoD,index));
    }


    /*  @Test
      public void testShape() {
          INDArrayIndex[] pointIndexes = {NDArrayIndex.point(0)};
          int[] shapeTest = Indices.shape(shape,pointIndexes);
          //drop a shape
          assertArrayEquals(new int[]{1,2,1,3,4,5,1},shapeTest);
          //keep the same shape
          assertArrayEquals(shape,Indices.shape(shape,new INDArrayIndex[]{NDArrayIndex.interval(0,2)}));
          //prepend an index
          assertArrayEquals(new int[]{1,1,2,3,4,5},Indices.shape(shape,new INDArrayIndex[]{NDArrayIndex.newAxis()}));
      }

      @Test
      public void test2dCases() {
          int[] shape = {3,2};
          assertArrayEquals(new int[]{1,2},Indices.shape(shape,new INDArrayIndex[]{NDArrayIndex.point(1)}));
          assertArrayEquals(new int[]{1,1},Indices.shape(shape,new INDArrayIndex[]{NDArrayIndex.interval(0,1)}));
       assertArrayEquals(new int[]{1,3,2},Indices.shape(shape,new INDArrayIndex[]{NDArrayIndex.newAxis()}));
      }
  */
    @Override
    public char ordering() {
        return 'f';
    }
}
