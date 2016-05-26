package org.nd4j.linalg.dataset;


import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Collections;
import java.util.List;

/**
 * Created by susaneraly on 5/25/16.
 */
@RunWith(Parameterized.class)
public class NormalizerStandardizeTest {
    public NormalizerStandardizeTest(Nd4jBackend backend) {
        super(backend);
    }
    @Test
    public void testMeanStd() {
        // use rand to generate an ndarray with mean 0 and std 1
        DataSet randNDataSet;
        BaseDatasetIterator randNIterator;
        NormalizerStandardize myNormalizer = new NormalizerStandardize();
        int nSamples = 7500;
        int nFeatures = 4;
        int bSize = 1;
        int i = nFeatures;
        int a = 5;
        int b = 6;
        // use randn to generate an ndarray with mean 0 and std 1
        // transform ndarray as X = A + BX
        // aA and bB are generated from a/b as a uniformly distr vector
        INDArray aA = Nd4j.rand(1,nFeatures).mul(a);
        INDArray bB = Nd4j.rand(1,nFeatures).mul(b);
        randNDataSet = randomDataSet(nSamples,nFeatures,aA,bB);
        myNormalizer.fit(randNDataSet);
        INDArray actualMean = myNormalizer.getMean();
        INDArray actualStd = myNormalizer.getStd();

        randNIterator = new BaseDatasetIterator(bSize,nSamples,randNDataSet.asList());
        randNIterator.setPreProcessor(myNormalizer);
        // assert mean is a
        // asset std dev is sqrt(b)
    }
    private DataSet randomDataSet(int nSamples, int nFeatures, INDArray aA, INDArray bB) {
        int i = 0;
        INDArray randomFeatures = Nd4j.zeros(nSamples,nFeatures).add(aA);
        while (i < nFeatures) {
            INDArray randomSlice = Nd4j.randn(nSamples,1).mul(bB.getScalar(0,i));
            randomFeatures.putColumn(i,randomSlice);
        }
        INDArray randomLabels = Nd4j.zeros(nSamples,1);
        return new DataSet(randomFeatures,randomLabels);
    }
    @Override
    public char ordering() {
        return 'c';
    }
}
