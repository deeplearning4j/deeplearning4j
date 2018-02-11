package org.deeplearning4j.largevis;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;

public class LargeVisTest {

    @Test
    public void testLargeVisRun() {
        DataSet iris = new IrisDataSetIterator(150,150).next();
        LargeVis largeVis = LargeVis.builder()
                .vec(iris.getFeatureMatrix())
                .normalize(true)
                .seed(42).build();
        largeVis.fit();
    }

}
