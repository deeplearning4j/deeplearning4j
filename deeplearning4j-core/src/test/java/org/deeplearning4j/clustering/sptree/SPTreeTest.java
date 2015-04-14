package org.deeplearning4j.clustering.sptree;

import static org.junit.Assert.*;

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

        SpTree tree = new SpTree(3,data,2);
        assertEquals(Nd4j.create(new double[]{2.5,3.5,4.5}),tree.getCenterOfMass());
        assertFalse(tree.isLeaf());
        assertEquals(2, tree.getCumSize());
        assertEquals(8, tree.getNumChildren());
        assertTrue(tree.isCorrect());
        assertEquals(Nd4j.create(new double[]{4, 5, 6}), tree.getChildren()[0].getCenterOfMass());
        assertEquals(Nd4j.create(new double[]{2.5, 3.5, 4.5}), tree.getBoundary().corner());
        assertEquals(Nd4j.create(new double[]{1.5,1.5,1.5}),tree.getBoundary().width());
        assertTrue(tree.getChildren()[0].isLeaf());


    }
}
