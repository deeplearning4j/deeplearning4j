/* ******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.deeplearning4j.integration.testcases;


import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.*;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMLayerWeights;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.CompositeMultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.MultiDataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.resources.Resources;
import org.nd4j.shade.guava.io.Files;


import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class LSTMLayerTestCase {

    private static MultiNormalizerStandardize normalizer;

    public static void main(String[] args) throws Exception {

        SameDiff sd = getConfigurationTestCase1();
        sd.output(getTrainingData(),"out");
        sd = getConfigurationTestCase2();
        sd.output(getTrainingData(),"out");
        sd = getConfigurationTestCase3();
        sd.output(getTrainingData(),"out");



    }




    public static SameDiff getConfigurationTestCase1() {

        int miniBatchSize = 10;
        int numLabelClasses = 6;
        int numUnits = 10;
        int maxTSLength = 64;
        int sL = 10; //small just for test

        SameDiff sd = SameDiff.create();

        // notations:
        // bS - batch size
        // sL - sequence length, number of time steps
        // nIn - input size
        // nOut - output size (hidden size)


        // 2) [bS, sL, nIn]  when dataFormat == 2 (NST)

        SDVariable in = sd.placeHolder("in", DataType.FLOAT, miniBatchSize, sL, numUnits);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, miniBatchSize, numLabelClasses);


        SDVariable cLast = sd.var("cLast", Nd4j.zeros(DataType.FLOAT, miniBatchSize, numUnits));
        SDVariable yLast = sd.var("yLast", Nd4j.zeros(DataType.FLOAT, miniBatchSize, numUnits));


        SDVariable out = sd.rnn.lstmLayer(
                maxTSLength, in, cLast, yLast,
                LSTMLayerWeights.builder()
                        .weights(sd.var("weights", Nd4j.rand(numUnits, 4 * numUnits)))
                        .rWeights(sd.var("rWeights", Nd4j.rand(numUnits, 4 * numUnits)))
                        .peepholeWeights(sd.var("inputPeepholeWeights", Nd4j.rand(DataType.FLOAT, 3 * numUnits)))
                        .bias(sd.var("bias", Nd4j.rand(DataType.FLOAT, 4 * numUnits))).build(),
                LSTMLayerConfig.builder()
                        .lstmdataformat(LSTMDataFormat.NST)
                        .directionMode(LSTMDirectionMode.FWD)
                        .gateAct(LSTMActivations.SIGMOID)
                        .cellAct(LSTMActivations.TANH)
                        .outAct(LSTMActivations.TANH)
                        .retFullSequence(true)
                        .retLastC(true)
                        .retLastH(true)
                        .build()
        ).getLastOutput();

        //Also set the training configuration:
        sd.setTrainingConfig(TrainingConfig.builder()
                .dataSetFeatureMapping("in")            //features[0] -> "in" placeholder
                .dataSetLabelMapping("label")           //labels[0]   -> "label" placeholder
                .build());

        return sd;

    }



    public static SameDiff getConfigurationTestCase2() {

        int miniBatchSize = 10;
        int numLabelClasses = 6;
        int numUnits = 10;
        int maxTSLength = 64;
        int sL = 10; //small just for test



        SameDiff sd = SameDiff.create();
        // [sL, bS, nIn]  when dataFormat == TNS
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, sL, miniBatchSize, numUnits);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, miniBatchSize, numLabelClasses);


        SDVariable cLast = sd.var("cLast", Nd4j.zeros(DataType.FLOAT, miniBatchSize, numUnits));
        SDVariable yLast = sd.var("yLast", Nd4j.zeros(DataType.FLOAT, miniBatchSize, numUnits));


        SDVariable out = sd.rnn.lstmLayer(
                maxTSLength, in, cLast, yLast,
                LSTMLayerWeights.builder()
                        .weights(sd.var("weights", Nd4j.rand(numUnits, 4 * numUnits)))
                        .rWeights(sd.var("rWeights", Nd4j.rand(numUnits, 4 * numUnits)))
                         .build(),
                LSTMLayerConfig.builder()
                        .lstmdataformat(LSTMDataFormat.TNS)
                        .directionMode(LSTMDirectionMode.FWD)
                        .gateAct(LSTMActivations.SIGMOID)
                        .cellAct(LSTMActivations.TANH)
                        .outAct(LSTMActivations.TANH)
                        .retFullSequence(true)
                        .retLastC(false)
                        .retLastH(false)
                        .build()
        ).getOutput();

        //Also set the training configuration:
        sd.setTrainingConfig(TrainingConfig.builder()
                .dataSetFeatureMapping("in")            //features[0] -> "in" placeholder
                .dataSetLabelMapping("label")           //labels[0]   -> "label" placeholder
                .build());

        return sd;

    }


    public static SameDiff getConfigurationTestCase3() {

        int miniBatchSize = 10;
        int numLabelClasses = 6;
        int numUnits = 10;
        int maxTSLength = 64;
        int sL = 10; //small just for test



        SameDiff sd = SameDiff.create();



        // [bS, sL, nIn] when dataFormat == NTS
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, miniBatchSize,sL, numUnits);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, miniBatchSize, numLabelClasses);





        // when directionMode >= 2 (BIDIR_CONCAT=3)
        // Wx, Wr [2, nIn, 4*nOut]
        // hI, cI [2, bS, nOut]
        SDVariable cLast = sd.var("cLast", Nd4j.zeros(DataType.FLOAT, 2,miniBatchSize, numUnits));
        SDVariable yLast = sd.var("yLast", Nd4j.zeros(DataType.FLOAT, 2, miniBatchSize, numUnits));
        SDVariable out = sd.rnn.lstmLayer("out",
                maxTSLength, in, cLast, yLast,
                LSTMLayerWeights.builder()
                        .weights(sd.var("weights", Nd4j.rand(2,numUnits, 4 * numUnits)))
                        .rWeights(sd.var("rWeights", Nd4j.rand(2, numUnits, 4 * numUnits)))
                        .build(),
                LSTMLayerConfig.builder()
                        .lstmdataformat(LSTMDataFormat.NTS)
                        .directionMode(LSTMDirectionMode.BIDIR_CONCAT)
                        .gateAct(LSTMActivations.SIGMOID)
                        .cellAct(LSTMActivations.SOFTPLUS)
                        .outAct(LSTMActivations.SOFTPLUS)
                        .retFullSequence(true)
                        .retLastC(false)
                        .retLastH(false)
                        .build()
        ).getOutput();

        //Also set the training configuration:
        sd.setTrainingConfig(TrainingConfig.builder()
                .dataSetFeatureMapping("in")            //features[0] -> "in" placeholder
                .dataSetLabelMapping("label")           //labels[0]   -> "label" placeholder
                .build());

        return sd;

    }


    public static List<Map<String, INDArray>> getPredictionsTestDataSameDiff() throws Exception {

        MultiDataSet mds = getTrainingData().next();

        List<Map<String, INDArray>> list = new ArrayList<>();

        list.add(Collections.singletonMap("in", mds.getFeatures()[0].reshape(10, 60)));
        //[batchsize, insize]
        System.out.println(mds.getFeatures()[0].reshape(10, 60));

        return list;
    }


    public List<String> getPredictionsNamesSameDiff() throws Exception {
        return Collections.singletonList("out");
    }


    public static MultiDataSetIterator getTrainingData() throws Exception {
        MultiDataSetIterator iter = getTrainingDataUnnormalized();
        MultiDataSetPreProcessor pp = multiDataSet -> {
            INDArray l = multiDataSet.getLabels(0);
            l = l.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(l.size(2) - 1));
            multiDataSet.setLabels(0, l);
            multiDataSet.setLabelsMaskArray(0, null);
        };


        iter.setPreProcessor(new CompositeMultiDataSetPreProcessor(getNormalizer(), pp));

        return iter;
    }

    protected static MultiDataSetIterator getTrainingDataUnnormalized() throws Exception {
        int miniBatchSize = 10;
        int numLabelClasses = 6;

        File featuresDirTrain = Files.createTempDir();
        File labelsDirTrain = Files.createTempDir();
        Resources.copyDirectory("dl4j-integration-tests/data/uci_seq/train/features/", featuresDirTrain);
        Resources.copyDirectory("dl4j-integration-tests/data/uci_seq/train/labels/", labelsDirTrain);

        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));

        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
                false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        MultiDataSetIterator iter = new MultiDataSetIteratorAdapter(trainData);

        return iter;
    }


    protected static MultiDataNormalization getNormalizer() throws Exception {
        if (normalizer != null) {
            return normalizer;
        }

        normalizer = new MultiNormalizerStandardize();
        normalizer.fit(getTrainingDataUnnormalized());

        return normalizer;
    }


    public MultiDataSetIterator getEvaluationTestData() throws Exception {
        int miniBatchSize = 10;
        int numLabelClasses = 6;

//            File featuresDirTest = new ClassPathResource("/RnnCsvSequenceClassification/uci_seq/test/features/").getFile();
//            File labelsDirTest = new ClassPathResource("/RnnCsvSequenceClassification/uci_seq/test/labels/").getFile();
        File featuresDirTest = Files.createTempDir();
        File labelsDirTest = Files.createTempDir();
        Resources.copyDirectory("dl4j-integration-tests/data/uci_seq/test/features/", featuresDirTest);
        Resources.copyDirectory("dl4j-integration-tests/data/uci_seq/test/labels/", labelsDirTest);

        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, 149));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, 149));

        DataSetIterator testData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
                false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        MultiDataSetIterator iter = new MultiDataSetIteratorAdapter(testData);

        MultiDataSetPreProcessor pp = multiDataSet -> {
            INDArray l = multiDataSet.getLabels(0);
            l = l.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(l.size(2) - 1));
            multiDataSet.setLabels(0, l);
            multiDataSet.setLabelsMaskArray(0, null);
        };


        iter.setPreProcessor(new CompositeMultiDataSetPreProcessor(getNormalizer(), pp));

        return iter;
    }
}













