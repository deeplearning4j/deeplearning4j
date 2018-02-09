package org.deeplearning4j.clustering.randomprojection;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

public class RPTreeTest {


    @Test
    public void testRPTree() throws Exception {
        DataSetIterator mnist = new IrisDataSetIterator(150,150);
        RPTree rpTree = new RPTree(4,50);
        DataSet d = mnist.next();
        NormalizerStandardize normalizerStandardize = new NormalizerStandardize();
        normalizerStandardize.fit(d);
        normalizerStandardize.transform(d.getFeatures());
        INDArray data = d.getFeatures();
        rpTree.buildTree(data);
        assertEquals(4,rpTree.getLeaves().size());
        assertEquals(0,rpTree.getRoot().getDepth());

        List<Integer> candidates = rpTree.getCandidates(data.getRow(0));
        assertFalse(candidates.isEmpty());
        assertEquals(10,rpTree.query(data.slice(0),10).length());
        System.out.println(candidates.size());

        rpTree.addNodeAtIndex(150,data.getRow(0));

    }

}
