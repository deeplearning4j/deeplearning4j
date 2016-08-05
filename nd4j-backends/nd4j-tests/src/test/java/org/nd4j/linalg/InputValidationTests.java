package org.nd4j.linalg;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by Alex on 05/08/2016.
 */
public class InputValidationTests {

    @Test(expected=IllegalStateException.class)
    public void testInvalidColVectorOp1(){
        INDArray first = Nd4j.create(10,10);
        INDArray col = Nd4j.create(5,1);
        first.muliColumnVector(col);
    }

    @Test(expected=IllegalStateException.class)
    public void testInvalidColVectorOp2(){
        INDArray first = Nd4j.create(10,10);
        INDArray col = Nd4j.create(5,1);
        first.addColumnVector(col);
    }

    @Test(expected=IllegalStateException.class)
    public void testInvalidRowVectorOp1(){
        INDArray first = Nd4j.create(10,10);
        INDArray row = Nd4j.create(1,5);
        first.addiRowVector(row);
    }

    @Test(expected=IllegalStateException.class)
    public void testInvalidRowVectorOp2(){
        INDArray first = Nd4j.create(10,10);
        INDArray row = Nd4j.create(1,5);
        first.subRowVector(row);
    }

}
