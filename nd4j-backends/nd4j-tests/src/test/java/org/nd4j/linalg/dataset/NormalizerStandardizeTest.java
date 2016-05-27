package org.nd4j.linalg.dataset;


import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;
import org.nd4j.linalg.dataset.api.iterator.BasicDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Collections;
import java.util.List;

/**
 * Created by susaneraly on 5/25/16.
 */
@RunWith(Parameterized.class)
public class NormalizerStandardizeTest extends BaseNd4jTest {
    public NormalizerStandardizeTest(Nd4jBackend backend) {
        super(backend);
    }
    @Test
    public void testMeanStd() {
        // use rand to generate an ndarray with mean 0 and std 1
        DataSet randNDataSet;
        DataSetIterator randNIterator;
        NormalizerStandardize myNormalizer = new NormalizerStandardize();
        int nSamples = 5000;
        int nFeatures = 3;
        int bSize = 1;
        int i = nFeatures;
        int a = 100;
        int b = 10;
        // use randn to generate an ndarray with mean 0 and std 1
        // transform ndarray as X = A + BX
        // aA and bB are generated from a/b as a uniformly distr vector
        INDArray aA = Nd4j.randn(1,nFeatures).mul(a);
        INDArray bB = Nd4j.randn(1,nFeatures).mul(b);
        System.out.println("===== aA and bB ======");
        System.out.println(aA);
        System.out.println(bB);
        System.out.println("===== Random Dataset ======");
        randNDataSet = randomDataSet(nSamples,nFeatures,aA,bB);
        System.out.println(randNDataSet);
        myNormalizer.fit(randNDataSet);
        System.out.println("======Actual mean and std =====");
        System.out.println(myNormalizer.getMean());
        System.out.println(myNormalizer.getStd());
        //Mean should br Aa+B*mean and stddev should be Bb
        INDArray actualMean = myNormalizer.getMean();
        INDArray actualStd = myNormalizer.getStd();
        DataSet blah = randNDataSet.copy();
        System.out.println("=====After being fit ======");
        myNormalizer.transform(blah);
        myNormalizer.fit(blah);
        System.out.println(blah);
        System.out.println("======Actual - Expected =====");
        System.out.println(actualMean.sub(aA));
        System.out.println(actualStd.sub(Transforms.abs(bB)));

        randNIterator = new BasicDataSetIterator(randNDataSet,bSize);
        myNormalizer.fit(randNIterator);
        randNIterator.setPreProcessor(myNormalizer);
        myNormalizer.transform(randNIterator);
        // assert mean is a
        // asset std dev is sqrt(b)
    }
    private DataSet randomDataSet(int nSamples, int nFeatures, INDArray aA, INDArray bB) {
        int i = 0;
        INDArray randomFeatures = Nd4j.zeros(nSamples,nFeatures);
        while (i < nFeatures) {
            INDArray randomSlice = Nd4j.randn(nSamples,1);
            randomSlice.muli(bB.getScalar(0,i));
            randomSlice.addi(aA.getScalar(0,i));
            randomFeatures.putColumn(i,randomSlice);
            i++;
        }
        INDArray randomLabels = Nd4j.zeros(nSamples,1);
        return new DataSet(randomFeatures,randomLabels);
    }
    @Override
    public char ordering() {
        return 'c';
    }
}
