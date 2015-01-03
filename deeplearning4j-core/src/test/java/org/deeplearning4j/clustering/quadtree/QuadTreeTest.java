package org.deeplearning4j.clustering.quadtree;
import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Set;

/**
 * Created by agibsonccc on 1/2/15.
 */
public class QuadTreeTest {

    @Test
    public void testQuadTree() {
        INDArray n = Nd4j.ones(3, 2);
        n.slice(1).addi(1);
        n.slice(2).addi(2);
        QuadTree quadTree = new QuadTree(n);
        //assertEquals(n.rows(),quadTree.getCumSize());
        Set<Integer> indices = quadTree.getIndices();
        assertEquals(n.rows(),indices.size());

    }

}
