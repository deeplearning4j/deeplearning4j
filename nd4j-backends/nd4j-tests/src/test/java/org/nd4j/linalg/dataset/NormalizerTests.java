package org.nd4j.linalg.dataset;

import org.junit.Before;
import org.junit.Test;
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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by susaneraly on 11/13/16.
 */
@RunWith(Parameterized.class)
public class NormalizerTests extends BaseNd4jTest {

    public NormalizerTests(Nd4jBackend backend) {
        super(backend);
    }

    private NormalizerStandardize stdScaler;
    private NormalizerMinMaxScaler minMaxScaler;
    private DataSet data;
    private int batchSize;
    private int batchCount;
    private int lastBatch;
    private final float thresholdPerc = 1.0f; //this is the difference in percentage!

    @Before
    public void randomData() {
        batchSize = 13;
        batchCount = 20;
        lastBatch = batchSize / 2;
        INDArray origFeatures = Nd4j.rand(batchCount * batchSize + lastBatch, 10);
        INDArray origLabels = Nd4j.rand(batchCount * batchSize + lastBatch, 3);
        data = new DataSet(origFeatures, origLabels);
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
        DataSetIterator dataIter = new TestDataSetIterator(dataCopy, batchSize);
        preProcessor.fit(dataCopy);
        preProcessor.transform(dataCopy);
        INDArray transformA = dataCopy.getFeatures();

        preProcessor.fit(dataIter);
        dataIter.setPreProcessor(preProcessor);
        DataSet next = dataIter.next();
        INDArray transformB = next.getFeatures();

        while (dataIter.hasNext()) {
            next = dataIter.next();
            INDArray transformb = next.getFeatures();
            transformB = Nd4j.vstack(transformB, transformb);
        }

        return Transforms.abs(transformB.div(transformA).rsub(1)).maxNumber().floatValue();
    }


    @Test
    public void testMasking(){

        DataNormalization[] normalizers = new DataNormalization[]{
                new NormalizerMinMaxScaler(),
                new NormalizerStandardize()};

        DataNormalization[] normalizersNoMask = new DataNormalization[]{
                new NormalizerMinMaxScaler(),
                new NormalizerStandardize()};


        for(int i=0; i<normalizers.length; i++ ){

            //First: check that normalization is the same with/without masking arrays
            DataNormalization norm = normalizers[i];
            DataNormalization normNoMask = normalizersNoMask[i];

            System.out.println(norm.getClass());


            INDArray arr = Nd4j.rand('c', new int[]{2, 3, 5}).muli(100).addi(100);
            arr.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.interval(3, 5)).assign(0);

            INDArray arrPt1 = arr.get(NDArrayIndex.interval(0,0,true), NDArrayIndex.all(), NDArrayIndex.all()).dup();
            INDArray arrPt2 = arr.get(NDArrayIndex.interval(1,1,true), NDArrayIndex.all(), NDArrayIndex.interval(0,3)).dup();


            INDArray mask = Nd4j.create(new double[][]{
                    {1, 1},
                    {1, 1},
                    {1, 1},
                    {1, 0},
                    {1, 0}});

            DataSet ds = new DataSet(arr, null, mask, null);
            norm.fit(ds);

            DataSet ds1 = new DataSet(arrPt1, null);
            DataSet ds2 = new DataSet(arrPt2, null);
            normNoMask.fit(ds1);
            normNoMask.fit(ds2);

            norm.transform(ds);
            normNoMask.transform(ds1);
            normNoMask.transform(ds2);

            assertEquals(ds1.getFeatureMatrix(), ds.getFeatureMatrix().get(NDArrayIndex.interval(0,0,true), NDArrayIndex.all(), NDArrayIndex.all()));
            assertEquals(ds2.getFeatureMatrix(), ds.getFeatureMatrix().get(NDArrayIndex.interval(1,1,true), NDArrayIndex.all(), NDArrayIndex.interval(0,3)));

            //Second: ensure values post masking are 0.0


            //Masked steps should be 0 after normalization
//            assertEquals(Nd4j.zeros(3, 2), arr.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.interval(3, 5)));
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
