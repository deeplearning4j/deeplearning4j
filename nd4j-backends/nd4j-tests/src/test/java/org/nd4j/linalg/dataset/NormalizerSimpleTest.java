package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.junit.Before;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by susaneraly on 11/13/16.
 */
@RunWith(Parameterized.class)
public class NormalizerSimpleTest  extends BaseNd4jTest {

    public NormalizerSimpleTest(Nd4jBackend backend) {
        super(backend);
    }

    private NormalizerStandardize stdScaler;
    private NormalizerMinMaxScaler minMaxScaler;
    private DataSet data;
    private int batchSize;
    private int batchCount;
    private int lastBatch;
    private DataSetIterator dataIter;
    private final float thresholdPerc = 0.01f; //this is the difference in percentage!

    @Before
    public void randomData() {
        batchSize = 13;
        batchCount = 7;
        lastBatch = batchSize/2;
        INDArray origFeatures = Nd4j.rand(batchCount*batchSize+lastBatch,10);
        INDArray origLabels = Nd4j.rand(batchCount*batchSize+lastBatch,3);
        data = new DataSet(origFeatures,origLabels);
        dataIter = new TestDataSetIterator(data,batchSize);
        stdScaler = new NormalizerStandardize();
        minMaxScaler = new NormalizerMinMaxScaler();
    }

    @Test
    public void testPreProcessors() {
        System.out.println("Running iterator vs non-iterator std scaler..");
        assertTrue(testItervsDataset(stdScaler) < thresholdPerc);
        System.out.println("Running iterator vs non-iterator min max scaler..");
        assertTrue(testItervsDataset(minMaxScaler) < thresholdPerc);
    }

    public float testItervsDataset(DataNormalization preProcessor) {
        DataSet dataCopy = data.copy();
        preProcessor.fit(dataCopy);
        preProcessor.transform(dataCopy);
        INDArray transformA = dataCopy.getFeatures();

        preProcessor.fit(dataIter);
        dataIter.setPreProcessor(preProcessor);
        DataSet next = dataIter.next();
        INDArray transformB = next.getFeatures();
        //INDArray transformb = transformB.dup();
        while (dataIter.hasNext()) {
            next = dataIter.next();
            INDArray transformb = next.getFeatures();
            transformB = Nd4j.vstack(transformB,transformb);
        }

        //System.out.println(transformb.size(0) + " should be "+ lastBatch);

        return Transforms.abs(transformB.div(transformA).rsub(1)).maxNumber().floatValue();
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
