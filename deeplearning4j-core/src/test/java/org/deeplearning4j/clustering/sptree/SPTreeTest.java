package org.deeplearning4j.clustering.sptree;

import static org.junit.Assert.*;
import static org.junit.Assume.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Adam Gibson
 */
public class SPTreeTest {

    @Test
    public void testStructure() {
        INDArray data = Nd4j.create(new double[][]{
                {1,2,3},{4,5,6}
        });

        SpTree tree = new SpTree(data);
        assertEquals(Nd4j.create(new double[]{2.5,3.5,4.5}),tree.getCenterOfMass());
        assertTrue(tree.isLeaf());
        assertEquals(2, tree.getCumSize());
        assertEquals(8, tree.getNumChildren());
        assertTrue(tree.isCorrect());

    }

    @Test
    public void testLargeTree() {
        int num = 10;
        INDArray arr = Nd4j.linspace(1,num,num).reshape(num,1);
        SpTree tree = new SpTree(arr);
    }

}
