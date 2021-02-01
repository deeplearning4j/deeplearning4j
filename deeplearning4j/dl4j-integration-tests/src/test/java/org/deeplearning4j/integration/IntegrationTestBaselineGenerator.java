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

package org.deeplearning4j.integration;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.MultiDataSetWrapperIterator;
import org.deeplearning4j.integration.testcases.samediff.SameDiffCNNCases;
import org.deeplearning4j.integration.testcases.samediff.SameDiffMLPTestCases;
import org.deeplearning4j.integration.testcases.samediff.SameDiffRNNTestCases;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.CollectScoresListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.common.base.Preconditions;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.nd4j.shade.guava.io.Files;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

/**
 * Run this manually to generate - or update - the saved files for a specific test.
 * Places results in dl4j-test-resources: assumes you have the dl4j-test-resources cloned parallel to the DL4J mono-repo.
 */
@Slf4j
public class IntegrationTestBaselineGenerator {

    public static final File OUTPUT_DIR_DL4J = new File("../../dl4j-test-resources/src/main/resources/dl4j-integration-tests").getAbsoluteFile();
    public static final File OUTPUT_DIR_SAMEDIFF = new File("../../dl4j-test-resources/src/main/resources/samediff-integration-tests").getAbsoluteFile();


    public static void main(String[] args) throws Exception {
        if (!OUTPUT_DIR_DL4J.exists() && !OUTPUT_DIR_SAMEDIFF.exists()) {
            throw new RuntimeException("output directories in test resources do not exist!");
        }

        runGeneration(

                //  DL4J integration test cases.

//                CNN1DTestCases.getCnn1dTestCaseCharRNN(),
//                CNN2DTestCases.testLenetTransferDropoutRepeatability(),
////                CNN2DTestCases.getCnn2DSynthetic(),
//                CNN2DTestCases.getLenetMnist(),
//                CNN2DTestCases.getVGG16TransferTinyImagenet(),
//                CNN2DTestCases.getYoloHouseNumbers(),
//                CNN3DTestCases.getCnn3dTestCaseSynthetic(),
//                MLPTestCases.getMLPMnist(),
//                MLPTestCases.getMLPMoon(),
//                RNNTestCases.getRnnCharacterTestCase(),
//                RNNTestCases.getRnnCsvSequenceClassificationTestCase1(),
//                RNNTestCases.getRnnCsvSequenceClassificationTestCase2(),
//                UnsupervisedTestCases.getVAEMnistAnomaly(),

                //   Samediff test cases done
                SameDiffMLPTestCases.getMLPMnist(),
                SameDiffMLPTestCases.getMLPMoon(),
                SameDiffCNNCases.getLenetMnist(),
                SameDiffCNNCases.getCnn3dSynthetic(),
                SameDiffRNNTestCases.getRnnCsvSequenceClassificationTestCase1()
        );

    }

    private static void runGeneration(TestCase... testCases) throws Exception {

        for (TestCase tc : testCases) {
            final ModelType modelType = tc.modelType();

            //Basic validation:
            Preconditions.checkState(tc.getTestName() != null, "Test case name is null");

            //Run through each test case:
            File testBaseDir = new File(modelType == ModelType.SAMEDIFF ? OUTPUT_DIR_SAMEDIFF : OUTPUT_DIR_DL4J, tc.getTestName());
            if (testBaseDir.exists()) {
                FileUtils.forceDelete(testBaseDir);
            }
            testBaseDir.mkdirs();

            File workingDir = Files.createTempDir();
            tc.initialize(workingDir);

            log.info("Starting result generation for test \"{}\" - output directory: {}", tc.getTestName(), testBaseDir.getAbsolutePath());

            //Step 0: collect metadata for the current machine, and write it (in case we need to debug anything related to
            // the comparison data)
            Properties properties = Nd4j.getExecutioner().getEnvironmentInformation();
            Properties pCopy = new Properties();
            String comment = System.getProperty("user.name") + " - " + System.currentTimeMillis();
//        StringBuilder sb = new StringBuilder(comment).append("\n");
            try (OutputStream os = new BufferedOutputStream(new FileOutputStream(new File(testBaseDir, "nd4jEnvironmentInfo.json")))) {
                Enumeration<Object> e = properties.keys();
                while (e.hasMoreElements()) {
                    Object k = e.nextElement();
                    Object v = properties.get(k);
                    pCopy.setProperty(k.toString(), v == null ? "null" : v.toString());
                }
                pCopy.store(os, comment);
            }


            //First: if test is a random init test: generate the config, and save it
            MultiLayerNetwork mln = null;
            ComputationGraph cg = null;
            SameDiff sd = null;
            Model m = null;
            if (tc.getTestType() == TestCase.TestType.RANDOM_INIT) {
                Object config = tc.getConfiguration();
                String json = null;
                if (config instanceof MultiLayerConfiguration) {
                    MultiLayerConfiguration mlc = (MultiLayerConfiguration) config;
                    json = mlc.toJson();
                    mln = new MultiLayerNetwork(mlc);
                    mln.init();
                    m = mln;
                } else if (config instanceof ComputationGraphConfiguration) {
                    ComputationGraphConfiguration cgc = (ComputationGraphConfiguration) config;
                    json = cgc.toJson();
                    cg = new ComputationGraph(cgc);
                    cg.init();
                    m = cg;
                } else {
                    sd = (SameDiff) config;
                }

                File savedModel = new File(testBaseDir, IntegrationTestRunner.RANDOM_INIT_UNTRAINED_MODEL_FILENAME);
                if (modelType != ModelType.SAMEDIFF) {
                    File configFile = new File(testBaseDir, "config." + (modelType == ModelType.MLN ? "mlc.json" : "cgc.json"));
                    FileUtils.writeStringToFile(configFile, json, StandardCharsets.UTF_8);
                    log.info("RANDOM_INIT test - saved configuration: {}", configFile.getAbsolutePath());
                    ModelSerializer.writeModel(m, savedModel, true);
                } else {
                    sd.save(savedModel, true);
                }
                log.info("RANDOM_INIT test - saved randomly initialized model to: {}", savedModel.getAbsolutePath());
            } else {
                //Pretrained model
                m = tc.getPretrainedModel();
                if (m instanceof MultiLayerNetwork) {
                    mln = (MultiLayerNetwork) m;
                } else if (m instanceof ComputationGraph) {
                    cg = (ComputationGraph) m;
                } else {
                    sd = (SameDiff) m;
                }
            }


            //Generate predictions to compare against
            if (tc.isTestPredictions()) {
                List<Pair<INDArray[], INDArray[]>> inputs = modelType != ModelType.SAMEDIFF ? tc.getPredictionsTestData() : null;
                List<Map<String, INDArray>> inputsSd = modelType == ModelType.SAMEDIFF ? tc.getPredictionsTestDataSameDiff() : null;
//                Preconditions.checkState(inputs != null && inputs.size() > 0, "Input data is null or length 0 for test: %s", tc.getTestName());


                File predictionsTestDir = new File(testBaseDir, "predictions");
                predictionsTestDir.mkdirs();

                int count = 0;
                if (modelType == ModelType.MLN) {
                    for (Pair<INDArray[], INDArray[]> p : inputs) {
                        INDArray f = p.getFirst()[0];
                        INDArray fm = (p.getSecond() == null ? null : p.getSecond()[0]);
                        INDArray out = mln.output(f, false, fm, null);

                        //Save the array...
                        File outFile = new File(predictionsTestDir, "output_" + (count++) + "_0.bin");
                        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(outFile))) {
                            Nd4j.write(out, dos);
                        }
                    }
                } else if (modelType == ModelType.CG) {
                    for (Pair<INDArray[], INDArray[]> p : inputs) {
                        INDArray[] out = cg.output(false, p.getFirst(), p.getSecond(), null);

                        //Save the array(s)...
                        for (int i = 0; i < out.length; i++) {
                            File outFile = new File(predictionsTestDir, "output_" + (count++) + "_" + i + ".bin");
                            try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(outFile))) {
                                Nd4j.write(out[i], dos);
                            }
                        }
                    }
                } else {
                    List<String> outNames = tc.getPredictionsNamesSameDiff();
                    for (Map<String, INDArray> ph : inputsSd) {
                        Map<String, INDArray> out = sd.output(ph, outNames);

                        //Save the output...
                        for (String s : outNames) {
                            File f = new File(predictionsTestDir, "output_" + (count++) + "_" + s + ".bin");
                            try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(f))) {
                                Nd4j.write(out.get(s), dos);
                            }
                        }
                    }
                }

                log.info("Saved predictions for {} inputs to disk in directory: {}", tc.getTestName(), predictionsTestDir);
            }

            //Compute and save gradients:
            if (tc.isTestGradients()) {
                INDArray gradientFlat = null;
                Map<String, INDArray> grad;
                if (modelType == ModelType.MLN) {
                    MultiDataSet data = tc.getGradientsTestData();
                    mln.setInput(data.getFeatures(0));
                    mln.setLabels(data.getLabels(0));
                    mln.setLayerMaskArrays(data.getFeaturesMaskArray(0), data.getLabelsMaskArray(0));
                    mln.computeGradientAndScore();
                    gradientFlat = mln.getFlattenedGradients();
                    grad = m.gradient().gradientForVariable();
                } else if (modelType == ModelType.CG) {
                    MultiDataSet data = tc.getGradientsTestData();
                    cg.setInputs(data.getFeatures());
                    cg.setLabels(data.getLabels());
                    cg.setLayerMaskArrays(data.getFeaturesMaskArrays(), data.getLabelsMaskArrays());
                    cg.computeGradientAndScore();
                    gradientFlat = cg.getFlattenedGradients();
                    grad = m.gradient().gradientForVariable();
                } else {
                    Map<String, INDArray> ph = tc.getGradientsTestDataSameDiff();
                    List<String> allVars = new ArrayList<>();
                    for (SDVariable v : sd.variables()) {
                        if (v.getVariableType() == VariableType.VARIABLE) {
                            allVars.add(v.name());
                        }
                    }
                    grad = sd.calculateGradients(ph, allVars);
                }

                if (modelType != ModelType.SAMEDIFF) {
                    File gFlatFile = new File(testBaseDir, IntegrationTestRunner.FLAT_GRADIENTS_FILENAME);
                    IntegrationTestRunner.write(gradientFlat, gFlatFile);
                }

                //Also save the gradient param table:
                File gradientDir = new File(testBaseDir, "gradients");
                gradientDir.mkdir();
                for (String s : grad.keySet()) {
                    File f = new File(gradientDir, s + ".bin");
                    IntegrationTestRunner.write(grad.get(s), f);
                }
            }

            //Test pretraining
            if (tc.isTestUnsupervisedTraining()) {
                log.info("Performing layerwise pretraining");
                MultiDataSetIterator iter = tc.getUnsupervisedTrainData();

                INDArray paramsPostTraining;
                if (modelType == ModelType.MLN) {
                    int[] layersToTrain = tc.getUnsupervisedTrainLayersMLN();
                    Preconditions.checkState(layersToTrain != null, "Layer indices must not be null");
                    DataSetIterator dsi = new MultiDataSetWrapperIterator(iter);

                    for (int i : layersToTrain) {
                        mln.pretrainLayer(i, dsi);
                    }
                    paramsPostTraining = mln.params();
                } else if (modelType == ModelType.CG) {
                    String[] layersToTrain = tc.getUnsupervisedTrainLayersCG();
                    Preconditions.checkState(layersToTrain != null, "Layer names must not be null");

                    for (String i : layersToTrain) {
                        cg.pretrainLayer(i, iter);
                    }
                    paramsPostTraining = cg.params();
                } else {
                    throw new UnsupportedOperationException("SameDiff not supported for unsupervised training tests");
                }

                //Save params
                File f = new File(testBaseDir, IntegrationTestRunner.PARAMS_POST_UNSUPERVISED_FILENAME);
                IntegrationTestRunner.write(paramsPostTraining, f);
            }

            //Test training curves:
            if (tc.isTestTrainingCurves()) {
                MultiDataSetIterator trainData = tc.getTrainingData();

                CollectScoresListener l = new CollectScoresListener(1);
                if (modelType != ModelType.SAMEDIFF)
                    m.setListeners(l);

                History h = null;
                if (modelType == ModelType.MLN) {
                    mln.fit(trainData);
                } else if (modelType == ModelType.CG) {
                    cg.fit(trainData);
                } else {
                    h = sd.fit(trainData, 1);
                }

                double[] scores;
                if (modelType != ModelType.SAMEDIFF) {
                    scores = l.getListScore().toDoubleArray();
                } else {
                    scores = h.lossCurve().getLossValues().toDoubleVector();
                }

                File f = new File(testBaseDir, IntegrationTestRunner.TRAINING_CURVE_FILENAME);
                List<String> s = Arrays.stream(scores).mapToObj(String::valueOf).collect(Collectors.toList());
                FileUtils.writeStringToFile(f, String.join(",", s), StandardCharsets.UTF_8);

                if (tc.isTestParamsPostTraining()) {
                    if (modelType == ModelType.SAMEDIFF) {
                        File p = new File(testBaseDir, IntegrationTestRunner.PARAMS_POST_TRAIN_SAMEDIFF_DIR);
                        p.mkdirs();
                        for (SDVariable v : sd.variables()) {
                            if (v.getVariableType() == VariableType.VARIABLE) {
                                INDArray arr = v.getArr();
                                File p2 = new File(p, v.name() + ".bin");
                                IntegrationTestRunner.write(arr, p2);
                            }
                        }
                    } else {
                        File p = new File(testBaseDir, IntegrationTestRunner.PARAMS_POST_TRAIN_FILENAME);
                        IntegrationTestRunner.write(m.params(), p);
                    }
                }
            }

            if (tc.isTestEvaluation()) {
                IEvaluation[] evals = tc.getNewEvaluations();
                MultiDataSetIterator iter = tc.getEvaluationTestData();

                if (modelType == ModelType.MLN) {
                    DataSetIterator dsi = new MultiDataSetWrapperIterator(iter);
                    mln.doEvaluation(dsi, evals);
                } else if (modelType == ModelType.CG) {
                    cg.doEvaluation(iter, evals);
                } else {
                    evals = tc.doEvaluationSameDiff(sd, iter, evals);
                }

                File evalDir = new File(testBaseDir, "evaluation");
                evalDir.mkdir();
                for (int i = 0; i < evals.length; i++) {
                    String json = evals[i].toJson();
                    File f = new File(evalDir, i + "." + evals[i].getClass().getSimpleName() + ".json");
                    FileUtils.writeStringToFile(f, json, StandardCharsets.UTF_8);
                }
            }

            //Don't need to do anything here re: overfitting
        }

        log.info("----- Completed test result generation -----");
    }
}
