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
package org.deeplearning4j.integration.testcases.samediff;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
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
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.resources.Resources;

import java.io.File;
import java.util.*;

import static org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig.*;

public class SameDiffMLPTestCases {


    public static TestCase getMLPMnist(){
        return new TestCase() {
            {
                testName = "MLPMnistSD";
                testType = TestType.RANDOM_INIT;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = true;
                testParamsPostTraining = true;
                testEvaluation = true;
                testOverfitting = true;
                maxRelativeErrorOverfit = 2e-2;
                minAbsErrorOverfit = 1e-2;
            }

            @Override
            public ModelType modelType() {
                return ModelType.SAMEDIFF;
            }

            @Override
            public Object getConfiguration() throws Exception {
                Nd4j.getRandom().setSeed(12345);

                //Define the network structure:
                SameDiff sd = SameDiff.create();
                SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 784);
                SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 10);

                SDVariable w0 = sd.var("w0", Nd4j.rand(DataType.FLOAT, 784, 256));
                SDVariable b0 = sd.var("b0", Nd4j.rand(DataType.FLOAT, 256));
                SDVariable w1 = sd.var("w1", Nd4j.rand(DataType.FLOAT, 256, 10));
                SDVariable b1 = sd.var("b1", Nd4j.rand(DataType.FLOAT, 10));

                SDVariable a0 = sd.nn.tanh(in.mmul(w0).add(b0));
                SDVariable out = sd.nn.softmax("out", a0.mmul(w1).add(b1));
                SDVariable loss = sd.loss.logLoss("loss", label, out);

                //Also set the training configuration:
                sd.setTrainingConfig(TrainingConfig.builder()
                        .updater(new Adam(0.01))
                        .weightDecay(1e-3, true)
                        .dataSetFeatureMapping("in")            //features[0] -> "in" placeholder
                        .dataSetLabelMapping("label")           //labels[0]   -> "label" placeholder
                        .build());

                return sd;
            }

            @Override
            public List<Map<String, INDArray>> getPredictionsTestDataSameDiff() throws Exception {
                List<Map<String,INDArray>> out = new ArrayList<>();

                DataSetIterator iter = new MnistDataSetIterator(1, true, 12345);
                out.add(Collections.singletonMap("in", iter.next().getFeatures()));

                iter = new MnistDataSetIterator(8, true, 12345);
                out.add(Collections.singletonMap("in", iter.next().getFeatures()));

                return out;
            }

            @Override
            public List<String> getPredictionsNamesSameDiff() throws Exception {
                return Collections.singletonList("out");
            }

            @Override
            public Map<String, INDArray> getGradientsTestDataSameDiff() throws Exception {
                DataSet ds = new MnistDataSetIterator(8, true, 12345).next();
                Map<String,INDArray> map = new HashMap<>();
                map.put("in", ds.getFeatures());
                map.put("label", ds.getLabels());
                return map;
            }

            @Override
            public MultiDataSetIterator getTrainingData() throws Exception {
                DataSetIterator iter = new MnistDataSetIterator(8, true, 12345);
                iter = new EarlyTerminationDataSetIterator(iter, 32);
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public IEvaluation[] getNewEvaluations() {
                return new IEvaluation[]{new Evaluation()};
            }

            @Override
            public MultiDataSetIterator getEvaluationTestData() throws Exception {
                DataSetIterator iter = new MnistDataSetIterator(8, false, 12345);
                iter = new EarlyTerminationDataSetIterator(iter, 32);
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public IEvaluation[] doEvaluationSameDiff(SameDiff sd, MultiDataSetIterator iter, IEvaluation[] evaluations) {
                sd.evaluate(iter, "out", 0, evaluations);
                return evaluations;
            }

            @Override
            public MultiDataSet getOverfittingData() throws Exception {
                return new MnistDataSetIterator(1, true, 12345).next().toMultiDataSet();
            }

            @Override
            public int getOverfitNumIterations() {
                return 100;
            }
        };
    }



    public static TestCase getMLPMoon() {
        return new TestCase() {
            {
                testName = "MLPMoonSD";
                testType = TestType.RANDOM_INIT;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = true;
                testParamsPostTraining = true;
                testEvaluation = true;
                testOverfitting = true;
                maxRelativeErrorOverfit = 2e-2;
                minAbsErrorOverfit = 1e-2;
            }

            @Override
            public ModelType modelType() {
                return ModelType.SAMEDIFF;
            }

            @Override
            public Object getConfiguration() throws Exception {

                int numInputs = 2;
                int numOutputs = 2;
                int numHiddenNodes = 20;
                double learningRate = 0.005;


                Nd4j.getRandom().setSeed(12345);

                //Define the network structure:
                SameDiff sd = SameDiff.create();
                SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, numInputs);
                SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, numOutputs);

                SDVariable w0 = sd.var("w0", Nd4j.rand(DataType.FLOAT, numInputs, numHiddenNodes));
                SDVariable b0 = sd.var("b0", Nd4j.rand(DataType.FLOAT, numHiddenNodes));
                SDVariable w1 = sd.var("w1", Nd4j.rand(DataType.FLOAT, numHiddenNodes, numOutputs));
                SDVariable b1 = sd.var("b1", Nd4j.rand(DataType.FLOAT, numOutputs));

                SDVariable a0 = sd.nn.relu(in.mmul(w0).add(b0), 0);
                SDVariable out = sd.nn.softmax("out", a0.mmul(w1).add(b1));
                SDVariable loss = sd.loss.logLoss("loss", label, out);

                //Also set the training configuration:
                sd.setTrainingConfig(TrainingConfig.builder()
                        .updater(new Nesterovs(learningRate, 0.9))
                        .weightDecay(1e-3, true)
                        .dataSetFeatureMapping("in")            //features[0] -> "in" placeholder
                        .dataSetLabelMapping("label")           //labels[0]   -> "label" placeholder
                        .build());

                return sd;
            }

            @Override
            public List<Map<String, INDArray>> getPredictionsTestDataSameDiff() throws Exception {
                List<Map<String, INDArray>> out = new ArrayList<>();

                File f = Resources.asFile("dl4j-integration-tests/data/moon_data_eval.csv");

                RecordReader rr = new CSVRecordReader();
                rr.initialize(new FileSplit(f));
                DataSetIterator iter = new RecordReaderDataSetIterator(rr, 1, 0, 2);

                out.add(Collections.singletonMap("in", iter.next().getFeatures()));


                return out;
            }


            @Override
            public List<String> getPredictionsNamesSameDiff() throws Exception {
                return Collections.singletonList("out");
            }

            @Override
            public Map<String, INDArray> getGradientsTestDataSameDiff() throws Exception {

                File f = Resources.asFile("dl4j-integration-tests/data/moon_data_eval.csv");
                RecordReader rr = new CSVRecordReader();
                rr.initialize(new FileSplit(f));
                org.nd4j.linalg.dataset.DataSet ds = new RecordReaderDataSetIterator(rr, 5, 0, 2).next();

                Map<String, INDArray> map = new HashMap<>();
                map.put("in", ds.getFeatures());
                map.put("label", ds.getLabels());
                return map;
            }

            @Override
            public MultiDataSetIterator getTrainingData() throws Exception {
                File f = new ClassPathResource("dl4j-integration-tests/data/moon_data_train.csv").getFile();
                RecordReader rr = new CSVRecordReader();
                rr.initialize(new FileSplit(f));
                DataSetIterator iter = new RecordReaderDataSetIterator(rr, 32, 0, 2);

                iter = new EarlyTerminationDataSetIterator(iter, 32);
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public IEvaluation[] getNewEvaluations() {
                return new IEvaluation[]{
                        new Evaluation(),
                        new ROCMultiClass(),
                        new EvaluationCalibration()};
            }

            @Override
            public MultiDataSetIterator getEvaluationTestData() throws Exception {
                File f = Resources.asFile("dl4j-integration-tests/data/moon_data_eval.csv");
                RecordReader rr = new CSVRecordReader();
                rr.initialize(new FileSplit(f));
                DataSetIterator iter = new RecordReaderDataSetIterator(rr, 32, 0, 2);
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public IEvaluation[] doEvaluationSameDiff(SameDiff sd, MultiDataSetIterator iter, IEvaluation[] evaluations) {
                sd.evaluate(iter, "out", 0, evaluations);
                return evaluations;
            }

            @Override
            public MultiDataSet getOverfittingData() throws Exception {

                File f = Resources.asFile("dl4j-integration-tests/data/moon_data_eval.csv");
                RecordReader rr = new CSVRecordReader();
                rr.initialize(new FileSplit(f));
                return new RecordReaderDataSetIterator(rr, 1, 0, 2).next().toMultiDataSet();
            }

            @Override
            public int getOverfitNumIterations() {
                return 200;
            }
        };



        public static TestCase getLenetMnist() {
            return new TestCase() {
                {
                    testName = "LenetMnistSD";
                    testType = TestType.RANDOM_INIT;
                    testPredictions = true;
                    testTrainingCurves = true;
                    testGradients = true;
                    testParamsPostTraining = true;
                    testEvaluation = true;
                    testOverfitting = false;
                }

                @Override
                public ModelType modelType() {
                    return ModelType.SAMEDIFF;
                }

                public Object getConfiguration() throws Exception {
                    int nChannels = 1; // Number of input channels
                    int outputNum = 10; // The number of possible outcomes
                    int seed = 123;

                    SameDiff sd = SameDiff.create();
                    SDVariable in = sd.placeHolder("in", DataType.FLOAT, nChannels, 784);
                    SDVariable label = sd.placeHolder("label", DataType.FLOAT, nChannels, outputNum);
                    SDVariable in4d = in.reshape(-1, nChannels, 28, 28);

                    int kernelHeight = 5;
                    int kernelWidth = 5;

                    //  for conv2d method: weight arrays must have shape:
                    //[kernelHeight, kernelWidth, inputChannels, outputChannels]
                    SDVariable w0 = sd.var("w0", Nd4j.rand(DataType.FLOAT, kernelHeight, kernelWidth, 20));

                    SDVariable layer0 = sd.cnn.conv2d("layer0", in4d, w0, Conv2DConfig.builder()
                            .kH(kernelHeight)
                            .kW(kernelWidth)
                            .build());

                    SDVariable w1 = sd.var("w1", Nd4j.rand(DataType.FLOAT, 20, 50));
                    SDVariable b1 = sd.var("b1", Nd4j.rand(DataType.FLOAT, 20));

                    SDVariable layer1 = sd.cnn.maxPooling2d("layer1", layer0.mmul(w1).add(b1),  Pooling2DConfig.builder()
                            .kH(2).kW(2)
                            .sH(2).sW(2)
                            .build());


                    SDVariable w2 = sd.var("w0", Nd4j.rand(DataType.FLOAT, kernelHeight, kernelWidth, 50));
                    SDVariable layer2 = sd.cnn.conv2d("layer2", layer1, w2, Conv2DConfig.builder()
                            .kH(kernelHeight)
                            .kW(kernelWidth)
                            .build());

                    SDVariable w3 = sd.var("w3", Nd4j.rand(DataType.FLOAT, 20, 50));
                    SDVariable b3 = sd.var("b3", Nd4j.rand(DataType.FLOAT, 20));

                    SDVariable layer3 = sd.cnn.maxPooling2d("layer1", layer2.mmul(w3).add(b3),  Pooling2DConfig.builder()
                            .kH(2).kW(2)
                            .sH(2).sW(2)
                            .build());

                    SDVariable w4 = sd.var("w4", Nd4j.rand(DataType.FLOAT, 50, 500));
                    SDVariable b4 = sd.var("b4", Nd4j.rand(DataType.FLOAT, 50));

                    SDVariable w5 = sd.var("w5", Nd4j.rand(DataType.FLOAT, 500, outputNum));
                    SDVariable b5 = sd.var("b5", Nd4j.rand(DataType.FLOAT, 500));

                    SDVariable layer4 = sd.nn.relu("layer4", layer3.mmul(w4).add(b4), 0);
                    SDVariable out = sd.nn.softmax("out", layer4.mmul(w5).add(b5));
                    SDVariable loss = sd.loss.logLoss("loss", label, out);


                    return sd;


                    //                Pooling2DConfig pooling2DConfig = new Pooling2DConfig();
//                pooling2DConfig.setType(Pooling2D.Pooling2DType.MAX);
//                pooling2DConfig.setKH(2);
//                pooling2DConfig.setKW(2);
//                pooling2DConfig.setSH(1);
//                pooling2DConfig.setSW(1);
//
//                SDVariable layer1 = sd.cnn.avgPooling2d("layer1",layer0,pooling2DConfig)
//
//
//                SDVariable a0 = sd.nn.relu(in.mmul(w0).add(b0),0);
//                SDVariable out = sd.nn.softmax("out", a0.mmul(w1).add(b1));
//                SDVariable loss = sd.loss.logLoss("loss", label, out);


//             STRUCTURE TO MAP
//                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                        .dataType(DataType.FLOAT)
//                        .seed(seed)
//                        .l2(0.0005)
//                        .weightInit(WeightInit.XAVIER)
//                        .updater(new Nesterovs(0.01, 0.9))
//                        .list()
//                        .layer(0, new ConvolutionLayer.Builder(5, 5)
//                                //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
//                                .nIn(nChannels)
//                                .stride(1, 1)
//                                .nOut(20)
//                                .activation(Activation.IDENTITY)
//                                .build())
//                        .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX)
//                                .kernelSize(2, 2)
//                                .stride(2, 2)
//                                .build())
//                        .layer(2, new ConvolutionLayer.Builder(5, 5)
//                                //Note that nIn need not be specified in later layers
//                                .stride(1, 1)
//                                .nOut(50)
//                                .activation(Activation.IDENTITY)
//                                .build())
//                        .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX)
//                                .kernelSize(2, 2)
//                                .stride(2, 2)
//                                .build())
//                        .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
//                                .nOut(500).build())
//                        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                                .nOut(outputNum)
//                                .activation(Activation.SOFTMAX)
//                                .build())
//                        .setInputType(InputType.convolutionalFlat(28, 28, 1)) //See note below
//                        .build();



                }

            }
        }
    }

}
