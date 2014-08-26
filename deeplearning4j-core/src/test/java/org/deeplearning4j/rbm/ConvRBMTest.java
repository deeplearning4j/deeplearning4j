package org.deeplearning4j.rbm;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.jblas.DoubleMatrix;
import org.junit.Ignore;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 4/21/14.
 */
public class ConvRBMTest {

    private static Logger log = LoggerFactory.getLogger(ConvRBMTest.class);

    @Test
    @Ignore
    public void testConvNet() throws Exception {
        ConvolutionalRBM rbm = new ConvolutionalRBM
                .Builder().withFilterSize(new int[]{7,7})
                .withNumFilters(new int[]{4,4}).withStride(new int[]{2,2})
                .withVisibleSize(new int[]{28,28}).numberOfVisible(28).numHidden(28).renderWeights(100)
                .build();

        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        fetcher.fetch(1);
        DataSet d = fetcher.next();
        INDArray train = d.getFeatureMatrix().reshape(28,28);

       rbm.trainTillConvergence(train,1e-2f,new Object[]{1,1e-2f,100});



    }

}
