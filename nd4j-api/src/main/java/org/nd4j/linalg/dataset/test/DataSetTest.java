package org.nd4j.linalg.dataset.test;

import static org.junit.Assert.*;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.NDArrays;
import org.nd4j.linalg.util.FeatureUtil;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Adam Gibson
 */
public abstract class DataSetTest {

    private static Logger log = LoggerFactory.getLogger(DataSetTest.class);

    @Test
    public void testFilterAndStrip() {
        INDArray labels = FeatureUtil.toOutcomeMatrix(new int[]{0,1,2,1,2,2,0,1,2,1},3);

        DataSet  d = new org.nd4j.linalg.dataset.DataSet(NDArrays.ones(10,2),labels);

        //strip the dataset down to just these labels. Rearrange them such that each label is in the specified position.
        d.filterAndStrip(new int[]{1,2});

        for(int i = 0; i < d.numExamples(); i++) {
            int outcome = d.get(i).outcome();
            assertTrue(outcome == 0 || outcome == 1);
        }



    }

}
