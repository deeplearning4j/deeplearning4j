package jcuda.jcublas.ops;

import org.apache.commons.math3.util.Pair;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static org.junit.Assert.assertTrue;

/**
 * Created by raver on 08.05.2016.
 */
public class ElementWiseStrideTests {

    @Test
    public void testEWS1() throws Exception {
        List<Pair<INDArray,String>> list = NDArrayCreationUtil.getAllTestMatricesWithShape(4,5,12345);
        list.addAll(NDArrayCreationUtil.getAll3dTestArraysWithShape(12345,4,5,6));
        list.addAll(NDArrayCreationUtil.getAll4dTestArraysWithShape(12345,4,5,6,7));
        list.addAll(NDArrayCreationUtil.getAll5dTestArraysWithShape(12345,4,5,6,7,8));
        list.addAll(NDArrayCreationUtil.getAll6dTestArraysWithShape(12345,4,5,6,7,8,9));


        for(Pair<INDArray,String> p : list){
            int ewsBefore = Shape.elementWiseStride(p.getFirst().shapeInfo());
            INDArray reshapeAttempt = Shape.newShapeNoCopy(p.getFirst(),new int[]{1,p.getFirst().length()}, Nd4j.order() == 'f');

            if (reshapeAttempt != null && ewsBefore == -1 && reshapeAttempt.elementWiseStride() != -1 ) {
                System.out.println("NDArrayCreationUtil." + p.getSecond());
                System.out.println("ews before: " + ewsBefore);
                System.out.println(p.getFirst().shapeInfoToString());
                System.out.println("ews returned by elementWiseStride(): " + p.getFirst().elementWiseStride());
                System.out.println("ews returned by reshape(): " + reshapeAttempt.elementWiseStride());
                System.out.println();
                assertTrue(false);
            } else {
          //      System.out.println("FAILED: " + p.getFirst().shapeInfoToString());
            }
        }
    }

    @Test
    public void testDualVStack() throws Exception {
        INDArray[] arrs = new INDArray[50];
        INDArray[] arrs2 = new INDArray[50];

        for( int i=0; i<arrs.length; i++ ){
            arrs[i] = Nd4j.create(new float[]{1f, 2f}).dup('c');
            arrs2[i] = Nd4j.create(new int[]{1,2},'c');
        }

        INDArray result = Nd4j.vstack(arrs);

        System.out.println("Result: " + result);
//        Nd4j.vstack(arrs2);
    }

    @Test
    public void testBVStack() throws Exception {
        INDArray[] arr = new INDArray[5];
        for( int i=0; i<arr.length; i++ ){
            arr[i] = Nd4j.create(new int[]{1,5749},'c');
        }

        Nd4j.vstack(arr);
        Nd4j.create(1);
    }

    @Test
    public void test2(){
        INDArray[] first = new INDArray[10];
        INDArray[] second = new INDArray[10];
        for( int i=0; i<10; i++ ){
            first[i] = Nd4j.create(new int[]{1,784},'c');
            second[i] = Nd4j.create(new int[]{1,5749},'c');
        }

        Nd4j.vstack(first);
        Nd4j.vstack(second);

        Nd4j.create(1);
    }
}
