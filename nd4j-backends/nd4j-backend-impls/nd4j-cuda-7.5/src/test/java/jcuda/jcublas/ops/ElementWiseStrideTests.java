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
}
