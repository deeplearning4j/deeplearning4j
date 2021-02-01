/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.integration.testcases.samediff;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.integration.ModelType;
import org.deeplearning4j.integration.TestCase;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.EvaluationCalibration;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMActivations;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDataFormat;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDirectionMode;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMLayerConfig;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.outputs.LSTMLayerOutputs;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMLayerWeights;
import org.nd4j.linalg.dataset.adapter.MultiDataSetIteratorAdapter;
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
import org.nd4j.common.resources.Resources;
import org.nd4j.shade.guava.io.Files;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class SameDiffRNNTestCases {

    public static TestCase getRnnCsvSequenceClassificationTestCase1() {
        return new SameDiffRNNTestCases.RnnCsvSequenceClassificationTestCase1();
    }

    protected static class RnnCsvSequenceClassificationTestCase1 extends TestCase {
        protected RnnCsvSequenceClassificationTestCase1() {
            testName = "RnnCsvSequenceClassification1";
            testType = TestType.RANDOM_INIT;
            testPredictions = true;
            testTrainingCurves = false;
            testGradients = false;
            testParamsPostTraining = false;
            testEvaluation = true;
            testOverfitting = false;            //Not much point on this one - it already fits very well...
        }


        protected MultiDataNormalization normalizer;

        protected MultiDataNormalization getNormalizer() throws Exception {
            if (normalizer != null) {
                return normalizer;
            }

            normalizer = new MultiNormalizerStandardize();
            normalizer.fit(getTrainingDataUnnormalized());

            return normalizer;
        }


        @Override
        public ModelType modelType() {
            return ModelType.SAMEDIFF;
        }


        @Override
        public Object getConfiguration() throws Exception {
            Nd4j.getRandom().setSeed(12345);


            int miniBatchSize = 10;
            int numLabelClasses = 6;
            int nIn = 60;
            int numUnits = 7;
            int timeSteps = 3;


            SameDiff sd = SameDiff.create();

            SDVariable in = sd.placeHolder("in", DataType.FLOAT, miniBatchSize, timeSteps, nIn);
            SDVariable label = sd.placeHolder("label", DataType.FLOAT, miniBatchSize, numLabelClasses);


            SDVariable cLast = sd.var("cLast", Nd4j.zeros(DataType.FLOAT, miniBatchSize, numUnits));
            SDVariable yLast = sd.var("yLast", Nd4j.zeros(DataType.FLOAT, miniBatchSize, numUnits));

            LSTMLayerConfig c = LSTMLayerConfig.builder()
                    .lstmdataformat(LSTMDataFormat.NTS)
                    .directionMode(LSTMDirectionMode.FWD)
                    .gateAct(LSTMActivations.SIGMOID)
                    .cellAct(LSTMActivations.TANH)
                    .outAct(LSTMActivations.TANH)
                    .retFullSequence(true)
                    .retLastC(true)
                    .retLastH(true)
                    .build();

            LSTMLayerOutputs outputs = new LSTMLayerOutputs(sd.rnn.lstmLayer(
                    in, cLast, yLast, null,
                    LSTMLayerWeights.builder()
                            .weights(sd.var("weights", Nd4j.rand(DataType.FLOAT, nIn, 4 * numUnits)))
                            .rWeights(sd.var("rWeights", Nd4j.rand(DataType.FLOAT, numUnits, 4 * numUnits)))
                            .peepholeWeights(sd.var("inputPeepholeWeights", Nd4j.rand(DataType.FLOAT, 3 * numUnits)))
                            .bias(sd.var("bias", Nd4j.rand(DataType.FLOAT, 4 * numUnits)))
                            .build(),
                    c), c);


//           Behaviour with default settings: 3d (time series) input with shape
//          [miniBatchSize, vectorSize, timeSeriesLength] -> 2d output [miniBatchSize, vectorSize]
            SDVariable layer0 = outputs.getOutput();

            SDVariable layer1 = layer0.mean(1);

            SDVariable w1 = sd.var("w1", Nd4j.rand(DataType.FLOAT, numUnits, numLabelClasses));
            SDVariable b1 = sd.var("b1", Nd4j.rand(DataType.FLOAT, numLabelClasses));


            SDVariable out = sd.nn.softmax("out", layer1.mmul(w1).add(b1));
            SDVariable loss = sd.loss.logLoss("loss", label, out);

            //Also set the training configuration:
            sd.setTrainingConfig(TrainingConfig.builder()
                    .updater(new Adam(5e-2))
                    .l1(1e-3).l2(1e-3)
                    .dataSetFeatureMapping("in")            //features[0] -> "in" placeholder
                    .dataSetLabelMapping("label")           //labels[0]   -> "label" placeholder
                    .build());

            return sd;

        }


        @Override
        public List<Map<String, INDArray>> getPredictionsTestDataSameDiff() throws Exception {

            MultiDataSet mds = getTrainingData().next();

            List<Map<String, INDArray>> list = new ArrayList<>();

            list.add(Collections.singletonMap("in", mds.getFeatures()[0].reshape(10, 1, 60)));
            //[batchsize, insize]

            return list;
        }

        @Override
        public List<String> getPredictionsNamesSameDiff() throws Exception {
            return Collections.singletonList("out");
        }


        @Override
        public MultiDataSetIterator getTrainingData() throws Exception {
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

        protected MultiDataSetIterator getTrainingDataUnnormalized() throws Exception {
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

        @Override
        public IEvaluation[] getNewEvaluations() {
            return new IEvaluation[]{
                    new Evaluation(),
                    new ROCMultiClass(),
                    new EvaluationCalibration()
            };
        }

        @Override
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

        @Override
        public IEvaluation[] doEvaluationSameDiff(SameDiff sd, MultiDataSetIterator iter, IEvaluation[] evaluations) {
            sd.evaluate(iter, "out", 0, evaluations);
            return evaluations;
        }
    }


}













