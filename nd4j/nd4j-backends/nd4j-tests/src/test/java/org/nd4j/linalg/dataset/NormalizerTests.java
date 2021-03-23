/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.dataset;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.adapter.MultiDataSetIteratorAdapter;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImageMultiPreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.MultiDataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag(TagNames.NDARRAY_ETL)
@NativeTag
@Tag(TagNames.FILE_IO)
public class NormalizerTests extends BaseNd4jTestWithBackends {


    private NormalizerStandardize stdScaler;
    private NormalizerMinMaxScaler minMaxScaler;
    private DataSet data;
    private int batchSize;
    private int batchCount;
    private int lastBatch;
    private final float thresholdPerc = 2.0f; //this is the difference in percentage!

    @BeforeEach
    public void randomData() {
        Nd4j.getRandom().setSeed(12345);
        batchSize = 13;
        batchCount = 20;
        lastBatch = batchSize / 2;
        INDArray origFeatures = Nd4j.rand(batchCount * batchSize + lastBatch, 10);
        INDArray origLabels = Nd4j.rand(batchCount * batchSize + lastBatch, 3);
        data = new DataSet(origFeatures, origLabels);
        stdScaler = new NormalizerStandardize();
        minMaxScaler = new NormalizerMinMaxScaler();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPreProcessors(Nd4jBackend backend) {
        System.out.println("Running iterator vs non-iterator std scaler..");
        double d1 = testItervsDataset(stdScaler);
        assertTrue( d1 < thresholdPerc,d1 + " < " + thresholdPerc);
        System.out.println("Running iterator vs non-iterator min max scaler..");
        double d2 = testItervsDataset(minMaxScaler);
        assertTrue(d2 < thresholdPerc,d2 + " < " + thresholdPerc);
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



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMasking(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(235);

        DataNormalization[] normalizers =
                new DataNormalization[] {new NormalizerMinMaxScaler(), new NormalizerStandardize()};

        DataNormalization[] normalizersNoMask =
                new DataNormalization[] {new NormalizerMinMaxScaler(), new NormalizerStandardize()};

        DataNormalization[] normalizersByRow =
                new DataNormalization[] {new NormalizerMinMaxScaler(), new NormalizerStandardize()};


        for (int i = 0; i < normalizers.length; i++) {
            //First: check that normalization is the same with/without masking arrays
            DataNormalization norm = normalizers[i];
            DataNormalization normFitSubset = normalizersNoMask[i];
            DataNormalization normByRow = normalizersByRow[i];

            System.out.println(norm.getClass());


            INDArray arr = Nd4j.rand('c', new int[] {2, 3, 5}).muli(100).addi(100);
            arr.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.interval(3, 5)).assign(0);
            INDArray arrCopy = arr.dup();

            INDArray arrPt1 = arr.get(NDArrayIndex.interval(0, 0, true), NDArrayIndex.all(), NDArrayIndex.all()).dup();
            INDArray arrPt2 =
                    arr.get(NDArrayIndex.interval(1, 1, true), NDArrayIndex.all(), NDArrayIndex.interval(0, 3))
                            .dup();


            INDArray mask = Nd4j.create(new double[][] {{1, 1, 1, 1, 1}, {1, 1, 1, 0, 0}}).castTo(Nd4j.defaultFloatingPointType());

            DataSet ds = new DataSet(arr, null, mask, null);
            DataSet dsCopy1 = new DataSet(arr.dup(), null, mask, null);
            DataSet dsCopy2 = new DataSet(arr.dup(), null, mask, null);
            norm.fit(ds);

            //Check that values aren't modified by fit op
            assertEquals(arrCopy, arr);

            List<DataSet> toFitTimeSeries1Ex = new ArrayList<>();
            toFitTimeSeries1Ex.add(new DataSet(arrPt1, arrPt1));
            toFitTimeSeries1Ex.add(new DataSet(arrPt2, arrPt2));
            normFitSubset.fit(new TestDataSetIterator(toFitTimeSeries1Ex, 1));

            List<DataSet> toFitRows = new ArrayList<>();
            for (int j = 0; j < 5; j++) {
                INDArray row = arr.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.interval(j, j, true))
                        .transpose();
                assertTrue(row.isRowVector());
                toFitRows.add(new DataSet(row, row));
            }

            for (int j = 0; j < 3; j++) {
                INDArray row = arr.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.interval(j, j, true))
                        .transpose();
                assertTrue(row.isRowVector());
                toFitRows.add(new DataSet(row, row));
            }

            normByRow.fit(new TestDataSetIterator(toFitRows, 1));

            norm.transform(ds);
            normFitSubset.transform(dsCopy1);
            normByRow.transform(dsCopy2);

            assertEquals(ds.getFeatures(), dsCopy1.getFeatures());
            assertEquals(ds.getLabels(), dsCopy1.getLabels());
            assertEquals(ds.getFeaturesMaskArray(), dsCopy1.getFeaturesMaskArray());
            assertEquals(ds.getLabelsMaskArray(), dsCopy1.getLabelsMaskArray());

            assertEquals(ds, dsCopy1);
            assertEquals(ds, dsCopy2);

            //Second: ensure time steps post normalization (and post revert) are 0.0
            INDArray shouldBe0_1 = ds.getFeatures().get(NDArrayIndex.point(1), NDArrayIndex.all(),
                    NDArrayIndex.interval(3, 5));
            INDArray shouldBe0_2 = dsCopy1.getFeatures().get(NDArrayIndex.point(1), NDArrayIndex.all(),
                    NDArrayIndex.interval(3, 5));
            INDArray shouldBe0_3 = dsCopy2.getFeatures().get(NDArrayIndex.point(1), NDArrayIndex.all(),
                    NDArrayIndex.interval(3, 5));

            INDArray zeros = Nd4j.zeros(shouldBe0_1.shape());

//            for (int j = 0; j < 2; j++) {
//                System.out.println(ds.getFeatures().get(NDArrayIndex.point(j), NDArrayIndex.all(),
//                                NDArrayIndex.all()));
//                System.out.println();
//            }

            assertEquals(zeros, shouldBe0_1);
            assertEquals(zeros, shouldBe0_2);
            assertEquals(zeros, shouldBe0_3);

            //Check same thing after reverting:
            norm.revert(ds);
            normFitSubset.revert(dsCopy1);
            normByRow.revert(dsCopy2);
            shouldBe0_1 = ds.getFeatures().get(NDArrayIndex.point(1), NDArrayIndex.all(),
                    NDArrayIndex.interval(3, 5));
            shouldBe0_2 = dsCopy1.getFeatures().get(NDArrayIndex.point(1), NDArrayIndex.all(),
                    NDArrayIndex.interval(3, 5));
            shouldBe0_3 = dsCopy2.getFeatures().get(NDArrayIndex.point(1), NDArrayIndex.all(),
                    NDArrayIndex.interval(3, 5));

            assertEquals(zeros, shouldBe0_1);
            assertEquals(zeros, shouldBe0_2);
            assertEquals(zeros, shouldBe0_3);


        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNormalizerToStringHashCode(){
        //https://github.com/eclipse/deeplearning4j/issues/8565

        testNormalizer(new NormalizerMinMaxScaler());
        NormalizerMinMaxScaler n1 = new NormalizerMinMaxScaler();
        n1.fitLabel(true);
        testNormalizer(n1);

        testNormalizer(new NormalizerStandardize());
        NormalizerStandardize n2 = new NormalizerStandardize();
        n2.fitLabel(true);
        testNormalizer(n2);

        testNormalizer(new ImagePreProcessingScaler());
        ImagePreProcessingScaler n3 = new ImagePreProcessingScaler();
        n3.fitLabel(true);
        testNormalizer(n3);

        testNormalizer(new VGG16ImagePreProcessor());
        VGG16ImagePreProcessor n4 = new VGG16ImagePreProcessor();
        n4.fitLabel(true);
        testNormalizer(n4);
    }

    private static void testNormalizer(DataNormalization n){
        n.toString();
        n.hashCode();

        n.fit(new IrisDataSetIterator(30, 150));

        n.toString();
        n.hashCode();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiNormalizerToStringHashCode(){
        //https://github.com/eclipse/deeplearning4j/issues/8565

        testMultiNormalizer(new MultiNormalizerMinMaxScaler());
        MultiNormalizerMinMaxScaler n1 = new MultiNormalizerMinMaxScaler();
        n1.fitLabel(true);
        testMultiNormalizer(n1);

        testMultiNormalizer(new MultiNormalizerStandardize());
        MultiNormalizerStandardize n2 = new MultiNormalizerStandardize();
        n2.fitLabel(true);
        testMultiNormalizer(n2);

        testMultiNormalizer(new ImageMultiPreProcessingScaler(0));
    }

    private static void testMultiNormalizer(MultiDataNormalization n){
        n.toString();
        n.hashCode();

        n.fit(new MultiDataSetIteratorAdapter(new IrisDataSetIterator(30, 150)));

        n.toString();
        n.hashCode();
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
