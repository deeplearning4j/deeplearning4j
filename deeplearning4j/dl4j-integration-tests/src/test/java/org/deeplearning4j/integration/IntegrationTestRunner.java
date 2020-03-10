/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.deeplearning4j.integration;


import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.MultiDataSetWrapperIterator;
import org.deeplearning4j.integration.util.CountingMultiDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.CollectScoresListener;
import org.deeplearning4j.parallelism.ParallelInference;
import org.deeplearning4j.parallelism.inference.InferenceMode;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.rules.TemporaryFolder;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.base.Preconditions;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.classification.*;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.RelativeError;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.resources.Resources;
import org.nd4j.shade.guava.collect.ImmutableSet;
import org.nd4j.shade.guava.reflect.ClassPath;

import java.io.*;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

/**
 * Integration test runner.
 * Used to run the integration tests defined in the IntegrationTests class
 *
 * @author Alex Black
 */
@Slf4j
public class IntegrationTestRunner {

    public static final String RANDOM_INIT_UNTRAINED_MODEL_FILENAME = "Model_RANDOM_INIT_UNTRAINED.zip";
    public static final String FLAT_GRADIENTS_FILENAME = "flattenedGradients.bin";
    public static final String TRAINING_CURVE_FILENAME = "trainingCurve.csv";
    public static final String PARAMS_POST_TRAIN_FILENAME = "paramsPostTrain.bin";
    public static final String PARAMS_POST_TRAIN_SAMEDIFF_DIR = "paramsPostTrain";
    public static final String PARAMS_POST_UNSUPERVISED_FILENAME = "paramsPostUnsupervised.bin";

    public static final double MAX_REL_ERROR_SCORES = 1e-4;

    private static List<Class<?>> layerClasses = new ArrayList<>();
    private static List<Class<?>> preprocClasses = new ArrayList<>();
    private static List<Class<?>> graphVertexClasses = new ArrayList<>();
    private static List<Class<?>> evaluationClasses = new ArrayList<>();

    private static Map<Class<?>, Integer> layerConfClassesSeen = new HashMap<>();
    private static Map<Class<?>, Integer> preprocessorConfClassesSeen = new HashMap<>();
    private static Map<Class<?>, Integer> vertexConfClassesSeen = new HashMap<>();
    private static Map<Class<?>, Integer> evaluationClassesSeen = new HashMap<>();

    static {
        try {
            setup();
        } catch (Exception e){
            throw new RuntimeException(e);
        }
    }


    public static void setup() throws Exception {

        //First: discover all layers, preprocessors, etc

        ImmutableSet<ClassPath.ClassInfo> info;
        try {
            //Dependency note: this ClassPath class was added in Guava 14
            info = ClassPath.from(DifferentialFunctionClassHolder.class.getClassLoader())
                    .getTopLevelClassesRecursive("org.deeplearning4j");
        } catch (IOException e) {
            //Should never happen
            throw new RuntimeException(e);
        }

        for (ClassPath.ClassInfo c : info) {
            Class<?> clazz = Class.forName(c.getName());
            if (Modifier.isAbstract(clazz.getModifiers()) || clazz.isInterface())
                continue;

            if (isLayerConfig(clazz)) {
                layerClasses.add(clazz);
            } else if (isPreprocessorConfig(clazz)) {
                preprocClasses.add(clazz);
            } else if (isGraphVertexConfig(clazz)) {
                graphVertexClasses.add(clazz);
            } else if (isEvaluationClass(clazz)) {
                evaluationClasses.add(clazz);
            }
        }

        layerClasses.sort(Comparator.comparing(Class::getName));
        preprocClasses.sort(Comparator.comparing(Class::getName));
        graphVertexClasses.sort(Comparator.comparing(Class::getName));

        log.info("Found {} layers", layerClasses.size());
        log.info("Found {} preprocessors", preprocClasses.size());
        log.info("Found {} graph vertices", graphVertexClasses.size());
        log.info("Found {} IEvaluation classes", evaluationClasses.size());

        layerConfClassesSeen = new HashMap<>();
        preprocessorConfClassesSeen = new HashMap<>();
        vertexConfClassesSeen = new HashMap<>();
        evaluationClassesSeen = new HashMap<>();
    }

    public static void runTest(TestCase tc, TemporaryFolder testDir) throws Exception {
//        BaseDL4JTest.skipUnlessIntegrationTests();
        //Tests will ONLY be run if integration test profile is enabled.
        //This could alternatively be done via maven surefire configuration

        final ModelType modelType = tc.modelType();
        log.info("Starting test case: {} - type = {}", tc.getTestName(), modelType);
        long start = System.currentTimeMillis();

        File workingDir = testDir.newFolder();
        tc.initialize(workingDir);

        File testBaseDir = testDir.newFolder();
//        new ClassPathResource("dl4j-integration-tests/" + tc.getTestName()).copyDirectory(testBaseDir);
        Resources.copyDirectory((modelType == ModelType.SAMEDIFF ? "samediff-integration-tests/" : "dl4j-integration-tests/") + tc.getTestName(), testBaseDir);


        MultiLayerNetwork mln = null;
        ComputationGraph cg = null;
        SameDiff sd = null;
        Model m = null;
        if (tc.getTestType() == TestCase.TestType.RANDOM_INIT) {
            log.info("Checking RANDOM_INIT test case: saved model vs. initialized model");
            //Checking randomly initialized model:
            File savedModel = new File(testBaseDir, IntegrationTestRunner.RANDOM_INIT_UNTRAINED_MODEL_FILENAME);
            Object config = tc.getConfiguration();
            if (config instanceof MultiLayerConfiguration) {
                MultiLayerConfiguration mlc = (MultiLayerConfiguration) config;
                mln = new MultiLayerNetwork(mlc);
                mln.init();
                m = mln;

                MultiLayerNetwork loaded = MultiLayerNetwork.load(savedModel, true);
                assertEquals("Configs not equal", loaded.getLayerWiseConfigurations(), mln.getLayerWiseConfigurations());
                assertEquals("Params not equal", loaded.params(), mln.params());
                assertEquals("Param table not equal", loaded.paramTable(), mln.paramTable());
            } else if(config instanceof ComputationGraphConfiguration ){
                ComputationGraphConfiguration cgc = (ComputationGraphConfiguration) config;
                cg = new ComputationGraph(cgc);
                cg.init();
                m = cg;

                ComputationGraph loaded = ComputationGraph.load(savedModel, true);
                assertEquals("Configs not equal", loaded.getConfiguration(), cg.getConfiguration());
                assertEquals("Params not equal", loaded.params(), cg.params());
                assertEquals("Param table not equal", loaded.paramTable(), cg.paramTable());
            } else if(config instanceof SameDiff){
                sd = (SameDiff)config;
                SameDiff loaded = SameDiff.load(savedModel, true);

                assertSameDiffEquals(sd, loaded);
            } else {
                throw new IllegalStateException("Unknown configuration/model type: " + config.getClass());
            }
        } else {
            m = tc.getPretrainedModel();
            if (m instanceof MultiLayerNetwork) {
                mln = (MultiLayerNetwork) m;
            } else if(m instanceof ComputationGraph) {
                cg = (ComputationGraph) m;
            } else if(m instanceof SameDiff){
                sd = (SameDiff)m;
            } else {
                throw new IllegalStateException("Unknown model type: " + m.getClass());
            }
        }

        //Collect information for test coverage
        if(modelType != ModelType.SAMEDIFF) {
            collectCoverageInformation(m);
        }


        //Check network output (predictions)
        if (tc.isTestPredictions()) {
            log.info("Checking predictions: saved output vs. initialized model");


            List<Pair<INDArray[], INDArray[]>> inputs = modelType != ModelType.SAMEDIFF ? tc.getPredictionsTestData() : null;
            List<Map<String,INDArray>> inputsSd = modelType == ModelType.SAMEDIFF ? tc.getPredictionsTestDataSameDiff() : null;
            Preconditions.checkState(modelType == ModelType.SAMEDIFF || inputs != null && inputs.size() > 0, "Input data is null or length 0 for test: %s", tc.getTestName());


            File predictionsTestDir = new File(testBaseDir, "predictions");
            predictionsTestDir.mkdirs();

            int count = 0;
            if (modelType == ModelType.MLN) {
                for (Pair<INDArray[], INDArray[]> p : inputs) {
                    INDArray f = p.getFirst()[0];
                    INDArray fm = (p.getSecond() == null ? null : p.getSecond()[0]);
                    INDArray out = mln.output(f, false, fm, null);

                    //Load the previously saved array
                    File outFile = new File(predictionsTestDir, "output_" + (count++) + "_0.bin");
                    INDArray outSaved;
                    try (DataInputStream dis = new DataInputStream(new FileInputStream(outFile))) {
                        outSaved = Nd4j.read(dis);
                    }

                    INDArray predictionExceedsRE = exceedsRelError(outSaved, out, tc.getMaxRelativeErrorOutput(), tc.getMinAbsErrorOutput());
                    int countExceeds = predictionExceedsRE.sumNumber().intValue();
                    assertEquals("Predictions do not match saved predictions - output", 0, countExceeds);
                }
            } else if(modelType == ModelType.CG){
                for (Pair<INDArray[], INDArray[]> p : inputs) {
                    INDArray[] out = cg.output(false, p.getFirst(), p.getSecond(), null);

                    //Load the previously saved arrays
                    INDArray[] outSaved = new INDArray[out.length];
                    for (int i = 0; i < out.length; i++) {
                        File outFile = new File(predictionsTestDir, "output_" + (count++) + "_" + i + ".bin");
                        try (DataInputStream dis = new DataInputStream(new FileInputStream(outFile))) {
                            outSaved[i] = Nd4j.read(dis);
                        }
                    }

                    for( int i=0; i<outSaved.length; i++ ){
                        INDArray predictionExceedsRE = exceedsRelError(outSaved[i], out[i], tc.getMaxRelativeErrorOutput(), tc.getMinAbsErrorOutput());
                        int countExceeds = predictionExceedsRE.sumNumber().intValue();
                        assertEquals("Predictions do not match saved predictions - output " + i, 0, countExceeds);
                    }
                }
            } else {
                List<String> outNames = tc.getPredictionsNamesSameDiff();
                for( Map<String,INDArray> ph : inputsSd ){
                    Map<String,INDArray> out = sd.output(ph, outNames);

                    //Load the previously saved placeholder arrays
                    Map<String,INDArray> outSaved = new HashMap<>();
                    for(String s : outNames){
                        File f = new File(predictionsTestDir, "output_" + (count++) + "_" + s + ".bin");
                        try (DataInputStream dis = new DataInputStream(new FileInputStream(f))) {
                            outSaved.put(s, Nd4j.read(dis));
                        }
                    }

                    for(String s : outNames){
                        INDArray predictionExceedsRE = exceedsRelError(outSaved.get(s), out.get(s), tc.getMaxRelativeErrorOutput(), tc.getMinAbsErrorOutput());
                        int countExceeds = predictionExceedsRE.sumNumber().intValue();
                        assertEquals("Predictions do not match saved predictions - output \"" + s + "\"", 0, countExceeds);
                    }
                }
            }

            if(modelType != ModelType.SAMEDIFF) {
                checkLayerClearance(m);
            }
        }


        //Test gradients
        if (tc.isTestGradients()) {
            log.info("Checking gradients: saved output vs. initialized model");

            INDArray gradientFlat = null;
            org.deeplearning4j.nn.api.Layer[] layers = null;
            Map<String,INDArray> grad;
            if (modelType == ModelType.MLN) {
                MultiDataSet data = tc.getGradientsTestData();
                mln.setInput(data.getFeatures(0));
                mln.setLabels(data.getLabels(0));
                mln.setLayerMaskArrays(data.getFeaturesMaskArray(0), data.getLabelsMaskArray(0));
                mln.computeGradientAndScore();
                gradientFlat = mln.getFlattenedGradients();
                layers = mln.getLayers();
                grad = mln.gradient().gradientForVariable();
            } else if(modelType == ModelType.CG) {
                MultiDataSet data = tc.getGradientsTestData();
                cg.setInputs(data.getFeatures());
                cg.setLabels(data.getLabels());
                cg.setLayerMaskArrays(data.getFeaturesMaskArrays(), data.getLabelsMaskArrays());
                cg.computeGradientAndScore();
                gradientFlat = cg.getFlattenedGradients();
                layers = cg.getLayers();
                grad = cg.gradient().gradientForVariable();
            } else {
                Map<String,INDArray> ph = tc.getGradientsTestDataSameDiff();
                List<String> allVars = new ArrayList<>();
                for(SDVariable v : sd.variables()){
                    if(v.getVariableType() == VariableType.VARIABLE){
                        allVars.add(v.name());
                    }
                }
                grad = sd.calculateGradients(ph, allVars);
            }

            if(modelType != ModelType.SAMEDIFF) {
                File gFlatFile = new File(testBaseDir, IntegrationTestRunner.FLAT_GRADIENTS_FILENAME);
                INDArray gradientFlatSaved = read(gFlatFile);

                INDArray gradExceedsRE = exceedsRelError(gradientFlatSaved, gradientFlat, tc.getMaxRelativeErrorGradients(), tc.getMinAbsErrorGradients());
                int count = gradExceedsRE.sumNumber().intValue();
                if (count > 0) {
                    logFailedParams(20, "Gradient", layers, gradExceedsRE, gradientFlatSaved, gradientFlat);
                }
                assertEquals("Saved flattened gradients: not equal (using relative error)", 0, count);
            }

            //Load the gradient table:
            File gradientDir = new File(testBaseDir, "gradients");
            for (File f : gradientDir.listFiles()) {
                if (!f.isFile()) {
                    continue;
                }
                String key = f.getName();
                key = key.substring(0, key.length() - 4); //remove ".bin"
                INDArray loaded = read(f);
                INDArray now = grad.get(key);


                INDArray gradExceedsRE = exceedsRelError(loaded, now, tc.getMaxRelativeErrorGradients(), tc.getMinAbsErrorGradients());
                int count = gradExceedsRE.sumNumber().intValue();
                assertEquals("Gradients: not equal (using relative error) for parameter: " + key, 0, count);
            }
        }

        //Test layerwise pretraining
        if(tc.isTestUnsupervisedTraining()){
            log.info("Performing layerwise pretraining");
            MultiDataSetIterator iter = tc.getUnsupervisedTrainData();

            INDArray paramsPostTraining;
            org.deeplearning4j.nn.api.Layer[] layers;
            if(modelType == ModelType.MLN){
                int[] layersToTrain = tc.getUnsupervisedTrainLayersMLN();
                Preconditions.checkState(layersToTrain != null, "Layer indices must not be null");
                DataSetIterator dsi = new MultiDataSetWrapperIterator(iter);

                for( int i : layersToTrain){
                    mln.pretrainLayer(i, dsi);
                }
                paramsPostTraining = mln.params();
                layers = mln.getLayers();
            } else if(modelType == ModelType.CG) {
                String[] layersToTrain = tc.getUnsupervisedTrainLayersCG();
                Preconditions.checkState(layersToTrain != null, "Layer names must not be null");

                for( String i : layersToTrain){
                    cg.pretrainLayer(i, iter);
                }
                paramsPostTraining = cg.params();
                layers = cg.getLayers();
            } else {
                throw new UnsupportedOperationException("Unsupported layerwise pretraining not supported for SameDiff models");
            }

            File f = new File(testBaseDir, IntegrationTestRunner.PARAMS_POST_UNSUPERVISED_FILENAME);
            INDArray expParams = read(f);

            INDArray exceedsRelError = exceedsRelError(expParams, paramsPostTraining, tc.getMaxRelativeErrorPretrainParams(),
                    tc.getMinAbsErrorPretrainParams());
            int count = exceedsRelError.sumNumber().intValue();
            if(count > 0){
                logFailedParams(20, "Parameter", layers, exceedsRelError, expParams, paramsPostTraining);
            }
            assertEquals("Number of parameters exceeding relative error", 0, count);

            //Set params to saved ones - to avoid accumulation of roundoff errors causing later failures...
            m.setParams(expParams);
        }


        //Test training curves:
        if (tc.isTestTrainingCurves() || tc.isTestParamsPostTraining()) {
            MultiDataSetIterator trainData = tc.getTrainingData();
            boolean isTbptt;
            int tbpttLength;
            if(modelType == ModelType.MLN){
                isTbptt = mln.getLayerWiseConfigurations().getBackpropType() == BackpropType.TruncatedBPTT;
                tbpttLength = mln.getLayerWiseConfigurations().getTbpttFwdLength();
            } else if(modelType == ModelType.CG) {
                isTbptt = cg.getConfiguration().getBackpropType() == BackpropType.TruncatedBPTT;
                tbpttLength = cg.getConfiguration().getTbpttFwdLength();
            } else {
                isTbptt = false;
                tbpttLength = 0;
            }

            CountingMultiDataSetIterator countingIter = new CountingMultiDataSetIterator(trainData, isTbptt, tbpttLength);
            CollectScoresListener l = new CollectScoresListener(1);
            if(modelType != ModelType.SAMEDIFF) {
                m.setListeners(l);
            }

            int iterBefore;
            int epochBefore;
            int iterAfter;
            int epochAfter;

            Map<String,INDArray> frozenParamsBefore = modelType != ModelType.SAMEDIFF ? getFrozenLayerParamCopies(m) : getConstantCopies(sd);
            org.deeplearning4j.nn.api.Layer[] layers = null;
            History h = null;
            if (modelType == ModelType.MLN) {
                iterBefore = mln.getIterationCount();
                epochBefore = mln.getEpochCount();
                mln.fit(countingIter);
                iterAfter = mln.getIterationCount();
                epochAfter = mln.getEpochCount();
                layers = mln.getLayers();
            } else if(modelType == ModelType.CG){
                iterBefore = cg.getConfiguration().getIterationCount();
                epochBefore = cg.getConfiguration().getEpochCount();
                cg.fit(countingIter);
                iterAfter = cg.getConfiguration().getIterationCount();
                epochAfter = cg.getConfiguration().getEpochCount();
                layers = cg.getLayers();
            } else {
                iterBefore = sd.getTrainingConfig().getIterationCount();
                epochBefore = sd.getTrainingConfig().getEpochCount();
                h = sd.fit(countingIter, 1);
                iterAfter = sd.getTrainingConfig().getIterationCount();
                epochAfter = sd.getTrainingConfig().getEpochCount();
            }

            //Check that frozen params (if any) haven't changed during training:
            if(modelType == ModelType.SAMEDIFF) {
                checkConstants(frozenParamsBefore, sd);
            } else {
                checkFrozenParams(frozenParamsBefore, m);
            }

            //Validate the iteration and epoch counts - both for the net, and for the layers
            int newIters = countingIter.getCurrIter();
            assertEquals(iterBefore + newIters, iterAfter);
            assertEquals(epochBefore + 1, epochAfter);
            if(modelType != ModelType.SAMEDIFF) {
                validateLayerIterCounts(m, epochBefore + 1, iterBefore + newIters);
            }


            double[] scores;
            if(modelType == ModelType.SAMEDIFF){
                scores = h.lossCurve().getLossValues().toDoubleVector();
            } else {
                scores = l.getListScore().toDoubleArray();
            }

            File f = new File(testBaseDir, IntegrationTestRunner.TRAINING_CURVE_FILENAME);
            String[] s = FileUtils.readFileToString(f, StandardCharsets.UTF_8).split(",");

            if(tc.isTestTrainingCurves()) {
                assertEquals("Different number of scores", s.length, scores.length);

                boolean pass = true;
                for (int i = 0; i < s.length; i++) {
                    double exp = Double.parseDouble(s[i]);
                    double re = relError(exp, scores[i]);
                    if (re > MAX_REL_ERROR_SCORES) {
                        pass = false;
                        break;
                    }
                }
                if (!pass) {
                    fail("Scores differ: expected/saved: " + Arrays.toString(s) + "\nActual: " + Arrays.toString(scores));
                }
            }

            if (tc.isTestParamsPostTraining()) {
                if(modelType != ModelType.SAMEDIFF) {
                    File p = new File(testBaseDir, IntegrationTestRunner.PARAMS_POST_TRAIN_FILENAME);
                    INDArray paramsExp = read(p);
                    INDArray z = exceedsRelError(m.params(), paramsExp, tc.getMaxRelativeErrorParamsPostTraining(), tc.getMinAbsErrorParamsPostTraining());
                    int count = z.sumNumber().intValue();
                    if (count > 0) {
                        logFailedParams(20, "Parameter", layers, z, paramsExp, m.params());
                    }
                    assertEquals("Number of params exceeded max relative error", 0, count);
                } else {
                    File dir = new File(testBaseDir, IntegrationTestRunner.PARAMS_POST_TRAIN_SAMEDIFF_DIR);
                    for(SDVariable v : sd.variables()){
                        if(v.getVariableType() != VariableType.VARIABLE)
                            continue;
                        INDArray paramNow = v.getArr();
                        File paramFile = new File(dir, v.name() + ".bin");
                        INDArray exp = read(paramFile);
                        INDArray z = exceedsRelError(paramNow, exp, tc.getMaxRelativeErrorParamsPostTraining(), tc.getMinAbsErrorParamsPostTraining());
                        int count = z.sumNumber().intValue();
                        if (count > 0) {
                            logFailedParams(20, "Parameter: " + v.name(), layers, z, exp, paramNow);
                        }
                        assertEquals("Number of params exceeded max relative error for parameter: \"" + v.name() + "\"", 0, count);
                    }
                }
            }

            if(modelType != ModelType.SAMEDIFF) {
                checkLayerClearance(m);
            }
        }

        //Check evaluation:
        if (tc.isTestEvaluation()) {
            log.info("Testing evaluation");
            IEvaluation[] evals = tc.getNewEvaluations();
            MultiDataSetIterator iter = tc.getEvaluationTestData();

            if (modelType == ModelType.MLN) {
                DataSetIterator dsi = new MultiDataSetWrapperIterator(iter);
                mln.doEvaluation(dsi, evals);
            } else if(modelType == ModelType.CG){
                cg.doEvaluation(iter, evals);
            } else {
                evals = tc.doEvaluationSameDiff(sd, iter, evals);
            }

            File evalDir = new File(testBaseDir, "evaluation");
            for (int i = 0; i < evals.length; i++) {
                File f = new File(evalDir, i + "." + evals[i].getClass().getSimpleName() + ".json");
                String json = FileUtils.readFileToString(f, StandardCharsets.UTF_8);
                IEvaluation e;
                if (evals[i].getClass() == Evaluation.class) {
                    e = Evaluation.fromJson(json);
                } else if (evals[i].getClass() == RegressionEvaluation.class) {
                    e = RegressionEvaluation.fromJson(json, RegressionEvaluation.class);
                } else if (evals[i].getClass() == ROC.class) {
                    e = ROC.fromJson(json, ROC.class);
                } else if (evals[i].getClass() == ROCBinary.class) {
                    e = ROCBinary.fromJson(json, ROCBinary.class);
                } else if (evals[i].getClass() == ROCMultiClass.class) {
                    e = ROCMultiClass.fromJson(json, ROCMultiClass.class);
                } else if (evals[i].getClass() == EvaluationCalibration.class) {
                    e = EvaluationCalibration.fromJson(json, EvaluationCalibration.class);
                } else {
                    throw new RuntimeException("Unknown/not implemented evaluation type: " + evals[i].getClass());
                }


                assertEquals("Evaluation not equal: " + evals[i].getClass(), e, evals[i]);

                //Evaluation coverage information:
                evaluationClassesSeen.put(evals[i].getClass(), evaluationClassesSeen.getOrDefault(evals[i].getClass(), 0) + 1);

                if(modelType != ModelType.SAMEDIFF) {
                    checkLayerClearance(m);
                }
            }
        }

        //Check model serialization
        {
            log.info("Testing model serialization");

            File f = testDir.newFile();
            f.delete();

            if (modelType == ModelType.MLN) {
                ModelSerializer.writeModel(m, f, true);
                MultiLayerNetwork restored = MultiLayerNetwork.load(f, true);
                assertEquals(mln.getLayerWiseConfigurations(), restored.getLayerWiseConfigurations());
                assertEquals(mln.params(), restored.params());
            } else if(modelType == ModelType.CG){
                ModelSerializer.writeModel(m, f, true);
                ComputationGraph restored = ComputationGraph.load(f, true);
                assertEquals(cg.getConfiguration(), restored.getConfiguration());
                assertEquals(cg.params(), restored.params());
            } else {
                sd.save(f, true);
                SameDiff restored = SameDiff.load(f, true);
                assertSameDiffEquals(sd, restored);
            }

            System.gc();
        }


        //Check parallel inference
        if (modelType != ModelType.SAMEDIFF && tc.isTestParallelInference()) {

            List<Pair<INDArray[], INDArray[]>> inputs = tc.getPredictionsTestData();

            int numThreads = 2; //TODO allow customization of this?

            List<INDArray[]> exp = new ArrayList<>();
            for(Pair<INDArray[], INDArray[]> p : inputs){
                INDArray[] out;
                if(modelType == ModelType.MLN){
                    INDArray fm = p.getSecond() == null ? null : p.getSecond()[0];
                    out = new INDArray[]{mln.output(p.getFirst()[0], false, fm, null)};
                } else {
                    out = cg.output(false, p.getFirst(), p.getSecond(), null);
                }
                exp.add(out);
            }

            ParallelInference inf =
                    new ParallelInference.Builder(m)
                            .inferenceMode(InferenceMode.BATCHED)
                            .batchLimit(3)
                            .queueLimit(8)
                            .workers(numThreads)
                            .build();


            testParallelInference(inf, inputs, exp);

            inf.shutdown();
            inf = null;
            System.gc();
        }


        //Test overfitting single example
        if (tc.isTestOverfitting()) {
            log.info("Testing overfitting on single example");

            MultiDataSet toOverfit = tc.getOverfittingData();
            for (int i = 0; i < tc.getOverfitNumIterations(); i++) {
                if (modelType == ModelType.MLN) {
                    mln.fit(toOverfit);
                } else if(modelType == ModelType.CG){
                    cg.fit(toOverfit);
                } else {
                    sd.fit(toOverfit);
                }
            }

            //Check:
            INDArray[] output = null;
            Map<String,INDArray> outSd = null;
            if (modelType == ModelType.MLN) {
                mln.setLayerMaskArrays(toOverfit.getFeaturesMaskArray(0), null);
                output = new INDArray[]{mln.output(toOverfit.getFeatures(0))};
            } else if(modelType == ModelType.CG ){
                cg.setLayerMaskArrays(toOverfit.getFeaturesMaskArrays(), null);
                output = cg.output(toOverfit.getFeatures());
            } else {
                List<String> l = sd.getTrainingConfig().getDataSetFeatureMapping();
                Map<String,INDArray> phMap = new HashMap<>();
                int i=0;
                for(String s : l){
                    phMap.put(s, toOverfit.getFeatures(i++));
                }
                outSd = sd.output(phMap, tc.getPredictionsNamesSameDiff());
            }

            int n = modelType == ModelType.SAMEDIFF ? outSd.size() : output.length;
            for (int i = 0; i < n; i++) {
                INDArray out = modelType == ModelType.SAMEDIFF ? outSd.get(tc.getPredictionsNamesSameDiff().get(i)) : output[i];
                INDArray label = toOverfit.getLabels(i);

                INDArray z = exceedsRelError(out, label, tc.getMaxRelativeErrorOverfit(), tc.getMinAbsErrorOverfit());
                int count = z.sumNumber().intValue();
                if (count > 0) {
                    System.out.println(out);
                    System.out.println(label);
                    INDArray re = relativeError(out, label, tc.getMinAbsErrorOverfit());
                    System.out.println("Relative error:");
                    System.out.println(re);
                }
                assertEquals("Number of outputs exceeded max relative error", 0, count);
            }

            if(modelType != ModelType.SAMEDIFF) {
                checkLayerClearance(m);
            }
        }

        long end = System.currentTimeMillis();


        log.info("Completed test case {} in {} sec", tc.getTestName(), (end - start) / 1000L);
    }

    //Work out which layers, vertices etc we have seen - so we can (at the end of all tests) log our integration test coverage
    private static void collectCoverageInformation(Model m){
        boolean isMLN = (m instanceof MultiLayerNetwork);
        MultiLayerNetwork mln = (isMLN ? (MultiLayerNetwork)m : null);
        ComputationGraph cg = (!isMLN ? (ComputationGraph)m : null);

        //Collect layer coverage information:
        org.deeplearning4j.nn.api.Layer[] layers;
        if (isMLN) {
            layers = mln.getLayers();
        } else {
            layers = cg.getLayers();
        }
        for (org.deeplearning4j.nn.api.Layer l : layers) {
            Layer lConf = l.conf().getLayer();
            layerConfClassesSeen.put(lConf.getClass(), layerConfClassesSeen.getOrDefault(lConf.getClass(), 0) + 1);
        }

        //Collect preprocessor coverage information:
        Collection<InputPreProcessor> preProcessors;
        if (isMLN) {
            preProcessors = mln.getLayerWiseConfigurations().getInputPreProcessors().values();
        } else {
            preProcessors = new ArrayList<>();
            for (org.deeplearning4j.nn.conf.graph.GraphVertex gv : cg.getConfiguration().getVertices().values()) {
                if (gv instanceof LayerVertex) {
                    InputPreProcessor pp = ((LayerVertex) gv).getPreProcessor();
                    if (pp != null) {
                        preProcessors.add(pp);
                    }
                }
            }
        }
        for (InputPreProcessor ipp : preProcessors) {
            preprocessorConfClassesSeen.put(ipp.getClass(), preprocessorConfClassesSeen.getOrDefault(ipp.getClass(), 0) + 1);
        }

        //Collect vertex coverage information
        if (!isMLN) {
            for (org.deeplearning4j.nn.conf.graph.GraphVertex gv : cg.getConfiguration().getVertices().values()) {
                vertexConfClassesSeen.put(gv.getClass(), vertexConfClassesSeen.getOrDefault(gv.getClass(), 0) + 1);
            }
        }
    }


    private static void checkLayerClearance(Model m) {
        //Check that the input fields for all layers have been cleared
        org.deeplearning4j.nn.api.Layer[] layers;
        if (m instanceof MultiLayerNetwork) {
            layers = ((MultiLayerNetwork) m).getLayers();
        } else {
            layers = ((ComputationGraph) m).getLayers();
        }

        for (org.deeplearning4j.nn.api.Layer l : layers) {
            assertNull(l.input());
            assertNull(l.getMaskArray());
            if (l instanceof BaseOutputLayer) {
                BaseOutputLayer b = (BaseOutputLayer) l;
                assertNull(b.getLabels());
            }
        }


        if (m instanceof ComputationGraph) {
            //Also check the vertices:
            GraphVertex[] vertices = ((ComputationGraph) m).getVertices();
            for (GraphVertex v : vertices) {
                int numInputs = v.getNumInputArrays();
                INDArray[] arr = v.getInputs();
                if (arr != null) {
                    for (int i = 0; i < numInputs; i++) {
                        assertNull(arr[i]);
                    }
                }
            }
        }
    }

    private static void validateLayerIterCounts(Model m, int expEpoch, int expIter){
        //Check that the iteration and epoch counts - on the layers - are synced
        org.deeplearning4j.nn.api.Layer[] layers;
        if (m instanceof MultiLayerNetwork) {
            layers = ((MultiLayerNetwork) m).getLayers();
        } else {
            layers = ((ComputationGraph) m).getLayers();
        }

        for(org.deeplearning4j.nn.api.Layer l : layers){
            assertEquals("Epoch count", expEpoch, l.getEpochCount());
            assertEquals("Iteration count", expIter, l.getIterationCount());
        }
    }


    private static Map<String,INDArray> getFrozenLayerParamCopies(Model m){
        Map<String,INDArray> out = new LinkedHashMap<>();
        org.deeplearning4j.nn.api.Layer[] layers;
        if (m instanceof MultiLayerNetwork) {
            layers = ((MultiLayerNetwork) m).getLayers();
        } else {
            layers = ((ComputationGraph) m).getLayers();
        }

        for(org.deeplearning4j.nn.api.Layer l : layers){
            if(l instanceof FrozenLayer){
                String paramPrefix;
                if(m instanceof MultiLayerNetwork){
                    paramPrefix = l.getIndex() + "_";
                } else {
                    paramPrefix = l.conf().getLayer().getLayerName() + "_";
                }
                Map<String,INDArray> paramTable = l.paramTable();
                for(Map.Entry<String,INDArray> e : paramTable.entrySet()){
                    out.put(paramPrefix + e.getKey(), e.getValue().dup());
                }
            }
        }

        return out;
    }

    private static Map<String,INDArray> getConstantCopies(SameDiff sd){
        Map<String,INDArray> out = new HashMap<>();
        for(SDVariable v : sd.variables()){
            if(v.isConstant()){
                out.put(v.name(), v.getArr());
            }
        }
        return out;
    }

    public static void checkFrozenParams(Map<String,INDArray> copiesBeforeTraining, Model m){
        for(Map.Entry<String,INDArray> e : copiesBeforeTraining.entrySet()){
            INDArray actual = m.getParam(e.getKey());
            assertEquals(e.getKey(), e.getValue(), actual);
        }
    }

    public static void checkConstants(Map<String,INDArray> copiesBefore, SameDiff sd){
        for(Map.Entry<String,INDArray> e : copiesBefore.entrySet()){
            INDArray actual = sd.getArrForVarName(e.getKey());
            assertEquals(e.getKey(), e.getValue(), actual);
        }
    }

    public static void printCoverageInformation(){

        log.info("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");

        log.info("Layer coverage - classes seen:");
        for (Class<?> c : layerClasses) {
            if (layerConfClassesSeen.containsKey(c)) {
                log.info("Class seen {} times in tests: {}", layerConfClassesSeen.get(c), c.getName());
            }
        }

        log.info("Layer classes NOT seen in any tests:");
        for (Class<?> c : layerClasses) {
            if (!layerConfClassesSeen.containsKey(c)) {
                log.info("Class NOT seen in any tests: {}", c.getName());
            }
        }

        log.info("----------------------------------------------------------------------------------------------------");

        log.info("GraphVertex coverage - classes seen:");
        for (Class<?> c : graphVertexClasses) {
            if (vertexConfClassesSeen.containsKey(c)) {
                log.info("Preprocessor seen {} times in tests: {}", preprocessorConfClassesSeen.get(c), c.getName());
            }
        }

        log.info("GraphVertexcoverage - classes NOT seen:");
        for (Class<?> c : graphVertexClasses) {
            if (!vertexConfClassesSeen.containsKey(c)) {
                log.info("Preprocessor NOT seen in any tests: {}", c.getName());
            }
        }

        log.info("----------------------------------------------------------------------------------------------------");

        log.info("Preprocessor coverage - classes seen:");
        for (Class<?> c : preprocClasses) {
            if (preprocessorConfClassesSeen.containsKey(c)) {
                log.info("Preprocessor seen {} times in tests: {}", preprocessorConfClassesSeen.get(c), c.getName());
            }
        }

        log.info("Preprocessor coverage - classes NOT seen:");
        for (Class<?> c : preprocClasses) {
            if (!preprocessorConfClassesSeen.containsKey(c)) {
                log.info("Preprocessor NOT seen in any tests: {}", c.getName());
            }
        }

        log.info("----------------------------------------------------------------------------------------------------");


        log.info("Evaluation coverage - classes seen:");
        for (Class<?> c : evaluationClasses) {
            if (evaluationClassesSeen.containsKey(c)) {
                log.info("Evaluation class seen {} times in tests: {}", evaluationClassesSeen.get(c), c.getName());
            }
        }

        log.info("Evaluation coverage - classes NOT seen:");
        for (Class<?> c : evaluationClasses) {
            if (!evaluationClassesSeen.containsKey(c)) {
                log.info("Evaluation class NOT seen in any tests: {}", c.getName());
            }
        }

        log.info("----------------------------------------------------------------------------------------------------");
    }

    private static boolean isLayerConfig(Class<?> c) {
        return Layer.class.isAssignableFrom(c);
    }

    private static boolean isPreprocessorConfig(Class<?> c) {
        return InputPreProcessor.class.isAssignableFrom(c);
    }

    private static boolean isGraphVertexConfig(Class<?> c) {
        return GraphVertex.class.isAssignableFrom(c);
    }

    private static boolean isEvaluationClass(Class<?> c) {
        return IEvaluation.class.isAssignableFrom(c);
    }

    private static INDArray read(File f) {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(f)))) {
            return Nd4j.read(dis);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void write(INDArray arr, File f) {
        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(f)))) {
            Nd4j.write(arr, dos);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static double relError(double d1, double d2) {
        Preconditions.checkState(!Double.isNaN(d1), "d1 is NaN");
        Preconditions.checkState(!Double.isNaN(d2), "d2 is NaN");
        if (d1 == 0.0 && d2 == 0.0) {
            return 0.0;
        }

        return Math.abs(d1 - d2) / (Math.abs(d1) + Math.abs(d2));
    }

    private static INDArray exceedsRelError(INDArray first, INDArray second, double maxRel, double minAbs) {
//        INDArray z = Nd4j.createUninitialized(first.shape());
//        Op op = new BinaryMinimalRelativeError(first, second, z, maxRel, minAbs);
//        Nd4j.getExecutioner().exec(op);
//        return z;
        INDArray z = relativeError(first, second, minAbs);
        BooleanIndexing.replaceWhere(z, 0.0, Conditions.lessThan(maxRel));
        BooleanIndexing.replaceWhere(z, 1.0, Conditions.greaterThan(0.0));
        return z;
    }

    private static INDArray relativeError(INDArray first, INDArray second) {
        INDArray z = Nd4j.createUninitialized(first.shape());
        Op op = new RelativeError(first, second, z);
        Nd4j.getExecutioner().exec(op);
        return z;
    }

    private static INDArray relativeError(@NonNull INDArray a1, @NonNull INDArray a2, double minAbsError) {
        long numNaN1 = Nd4j.getExecutioner().exec(new MatchCondition(a1, Conditions.isNan(), Integer.MAX_VALUE)).getInt(0);
        long numNaN2 = Nd4j.getExecutioner().exec(new MatchCondition(a2, Conditions.isNan(), Integer.MAX_VALUE)).getInt(0);
        Preconditions.checkState(numNaN1 == 0, "Array 1 has NaNs");
        Preconditions.checkState(numNaN2 == 0, "Array 2 has NaNs");


//        INDArray isZero1 = a1.eq(0.0);
//        INDArray isZero2 = a2.eq(0.0);
//        INDArray bothZero = isZero1.muli(isZero2);

        INDArray abs1 = Transforms.abs(a1, true);
        INDArray abs2 = Transforms.abs(a2, true);
        INDArray absDiff = Transforms.abs(a1.sub(a2), false);

        //abs(a1-a2) < minAbsError ? 1 : 0
        INDArray greaterThanMinAbs = Transforms.abs(a1.sub(a2), false);
        BooleanIndexing.replaceWhere(greaterThanMinAbs, 0.0, Conditions.lessThan(minAbsError));
        BooleanIndexing.replaceWhere(greaterThanMinAbs, 1.0, Conditions.greaterThan(0.0));

        INDArray result = absDiff.divi(abs1.add(abs2));
        //Only way to have NaNs given there weren't any in original : both 0s
        BooleanIndexing.replaceWhere(result, 0.0, Conditions.isNan());
        //Finally, set to 0 if less than min abs error, or unchanged otherwise
        result.muli(greaterThanMinAbs);

//        double maxRE = result.maxNumber().doubleValue();
//        if(maxRE > MAX_REL_ERROR){
//            System.out.println();
//        }
        return result;
    }

    public static void testParallelInference(@NonNull ParallelInference inf, List<Pair<INDArray[],INDArray[]>> in, List<INDArray[]> exp) throws Exception {
        final INDArray[][] act = new INDArray[in.size()][0];
        final AtomicInteger counter = new AtomicInteger(0);
        final AtomicInteger failedCount = new AtomicInteger(0);

        for( int i=0; i<in.size(); i++ ){
            final int j=i;
            new Thread(new Runnable() {
                @Override
                public void run() {
                    try{
                        INDArray[] inMask = in.get(j).getSecond();
                        act[j] = inf.output(in.get(j).getFirst(), inMask);
                        counter.incrementAndGet();
                    } catch (Exception e){
                        e.printStackTrace();
                        failedCount.incrementAndGet();
                    }
                }
            }).start();
        }

        long start = System.currentTimeMillis();
        long current = System.currentTimeMillis();
        while(current < start + 20000 && failedCount.get() == 0 && counter.get() < in.size()){
            Thread.sleep(1000L);
        }

        assertEquals(0, failedCount.get());
        assertEquals(in.size(), counter.get());
        for( int i=0; i<in.size(); i++ ){
            INDArray[] e = exp.get(i);
            INDArray[] a = act[i];

            assertArrayEquals(e, a);
        }
    }


    public static void logFailedParams(int maxNumToPrintOnFailure, String prefix, org.deeplearning4j.nn.api.Layer[] layers, INDArray exceedsRelError, INDArray exp, INDArray act){
        long length = exceedsRelError.length();
        int logCount = 0;
        for(int i=0; i<length; i++ ){
            if(exceedsRelError.getDouble(i) > 0){
                double dExp = exp.getDouble(i);
                double dAct = act.getDouble(i);
                double re = relError(dExp, dAct);
                double ae = Math.abs(dExp - dAct);

                //Work out parameter key:
                long pSoFar = 0;
                String pName = null;
                for(org.deeplearning4j.nn.api.Layer l : layers){
                    long n = l.numParams();
                    if(pSoFar + n < i){
                        pSoFar += n;
                    } else {
                        for(Map.Entry<String,INDArray> e : l.paramTable().entrySet()){
                            pSoFar += e.getValue().length();
                            if(pSoFar >= i){
                                pName = e.getKey();
                                break;
                            }
                        }
                    }
                }

                log.info("{} {} ({}) failed: expected {} vs actual {} (RelativeError: {}, AbsError: {})", i, prefix, pName, dExp, dAct, re, ae);
                if(++logCount >= maxNumToPrintOnFailure){
                    break;
                }
            }
        }
    }

    public static void assertSameDiffEquals(SameDiff sd1, SameDiff sd2){
        assertEquals(sd1.variableMap().keySet(), sd2.variableMap().keySet());
        assertEquals(sd1.getOps().keySet(), sd2.getOps().keySet());
        assertEquals(sd1.inputs(), sd2.inputs());

        //Check constant and variable arrays:
        for(SDVariable v : sd1.variables()){
            String n = v.name();
            assertEquals(n, v.getVariableType(), sd2.getVariable(n).getVariableType());
            if(v.isConstant() || v.getVariableType() == VariableType.VARIABLE){
                INDArray a1 = v.getArr();
                INDArray a2 = sd2.getVariable(n).getArr();
                assertEquals(n, a1, a2);
            }
        }

        //Check ops:
        for(SameDiffOp o : sd1.getOps().values()){
            SameDiffOp o2 = sd2.getOps().get(o.getName());
            assertEquals(o.getOp().getClass(), o2.getOp().getClass());
        }
    }
}
