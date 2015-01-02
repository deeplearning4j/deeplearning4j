package org.deeplearning4j.clustering.kdtree;

import static org.junit.Assert.*;

import org.deeplearning4j.berkeley.Pair;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 1/1/15.
 */
public class KDTreeTest {
    private static Logger log = LoggerFactory.getLogger(KDTreeTest.class);
    @Test
    public void testTree() {
        KDTree tree = new KDTree(2);
        INDArray half = Nd4j.create(Nd4j.createBuffer(new double[]{0.5, 0.5}));
        INDArray one = Nd4j.create(Nd4j.createBuffer(new double[]{1, 1}));
        tree.insert(half);
        tree.insert(one);
        Pair<Double,INDArray> pair = tree.nn(Nd4j.create(Nd4j.createBuffer(new double[]{0.5,0.5})));
        assertEquals(half,pair.getSecond());
    }


}
