package org.deeplearning4j.util;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.*;
import org.junit.Test;

import java.util.List;

/**
 * Created by agibsonccc on 6/11/14.
 */
public class MovingWindowMatrixTest {
    private static Logger log = LoggerFactory.getLogger(MovingWindowMatrixTest.class);

    @Test
    public void testMovingWindow() {
        INDArray ones = NDArrays.ones(4, 4);
        MovingWindowMatrix m = new MovingWindowMatrix(ones,2,2);
        List<INDArray> windows = m.windows();
        assertEquals(4,windows.size());
        MovingWindowMatrix m2 = new MovingWindowMatrix(ones,2,2,true);
        List<INDArray> windowsRotate  = m2.windows();
        assertEquals(16,windowsRotate.size());


    }



}
