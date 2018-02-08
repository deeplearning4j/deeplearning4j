package org.deeplearning4j.clustering.randomprojection;

import org.deeplearning4j.datasets.iterators.impl.MnistDataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class RPTreeTest {


    @Test
    public void testRPTree() throws Exception {
        DataSetIterator mnist = new MnistDataSetIterator(1000,1000);
        RPTree rpTree = new RPTree(784,1000);
        rpTree.buildTree(mnist.next().getFeatureMatrix());

    }

}
