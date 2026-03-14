package org.eclipse.deeplearning4j.nd4j.linalg.custom;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.indexing.conditions.Conditions;
import static org.junit.jupiter.api.Assertions.*;

public class MatchDebugTest {
    @Test
    public void testMatchInt() {
        INDArray intArr = Nd4j.createFromArray(new int[]{0,0,0,1,0,2,2});
        System.err.println("INT dtype: " + intArr.dataType());
        INDArray mask = intArr.match(1, Conditions.equals(1));
        System.err.println("INT match result: " + mask);
        INDArray expected = Nd4j.createFromArray(false,false,false,true,false,false,false);
        assertEquals(expected, mask);
    }
    
    @Test
    public void testMatchDouble() {
        INDArray dblArr = Nd4j.createFromArray(new double[]{0,0,0,1,0,2,2});
        System.err.println("DOUBLE dtype: " + dblArr.dataType());
        INDArray mask = dblArr.match(1, Conditions.equals(1));
        System.err.println("DOUBLE match result: " + mask);
        INDArray expected = Nd4j.createFromArray(false,false,false,true,false,false,false);
        assertEquals(expected, mask);
    }
    
    @Test
    public void testMatchFloat() {
        INDArray fltArr = Nd4j.createFromArray(new float[]{0,0,0,1,0,2,2});
        System.err.println("FLOAT dtype: " + fltArr.dataType());
        INDArray mask = fltArr.match(1, Conditions.equals(1));
        System.err.println("FLOAT match result: " + mask);
        INDArray expected = Nd4j.createFromArray(false,false,false,true,false,false,false);
        assertEquals(expected, mask);
    }
    
    @Test
    public void testManualEq() {
        INDArray intArr = Nd4j.createFromArray(new int[]{0,0,0,1,0,2,2});
        INDArray eq = intArr.eq(1);
        System.err.println("Manual eq result: " + eq);
        INDArray expected = Nd4j.createFromArray(false,false,false,true,false,false,false);
        assertEquals(expected, eq);
    }
}
