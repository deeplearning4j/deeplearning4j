package org.deeplearning4j.integration;

import com.google.common.io.Files;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.MultiDataSetWrapperIterator;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.integration.testcases.*;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.CollectScoresListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Run this manually to generate - or update - the saved files for a specific test.
 * Places results in dl4j-test-resources: assumes you have the dl4j-test-resources cloned parallel to the DL4J mono-repo.
 */
@Slf4j
public class IntegrationTestBaselineGenerator {

    public static final File OUTPUT_DIR = new File("../../dl4j-test-resources/src/main/resources/dl4j-integration-tests").getAbsoluteFile();


    public static void main(String[] args) throws Exception {
        if (!OUTPUT_DIR.exists()) {
            throw new RuntimeException("output directory (test resources) does not exist!");
        }

        //All integration tests are run with float precision!
        Nd4j.setDataType(DataBuffer.Type.FLOAT);

//        runGeneration(
//                MLPTestCases.getMLPMnist(),
//                MLPTestCases.getMLPMoon(),
//                RNNTestCases.getRnnCsvSequenceClassificationTestCase1(),
//                RNNTestCases.getRnnCsvSequenceClassificationTestCase2(),
//                RNNTestCases.getRnnCharacterTestCase(),
////                CNN1DTestCases.getCnn1dTestCaseSynthetic(),
////                CNN2DTestCases.getLenetMnist(),
//                CNN2DTestCases.getVGG16TransferTinyImagenet(),
////                CNN2DTestCases.getYoloHouseNumbers(),
////                CNN2DTestCases.getCnn2DSynthetic(),
//                CNN2DTestCases.testLenetTransferDropoutRepeatability()//,
////                CNN3DTestCases.getCnn3dTestCaseSynthetic(),
////                UnsupervisedTestCases.getVAEMnistAnomaly(),
////                TransferLearningTestCases.testPartFrozenResNet50(),
////                TransferLearningTestCases.testPartFrozenNASNET()
//        );

        runGeneration(CNN2DTestCases.getCnn2DSynthetic(),
                CNN2DTestCases.getYoloHouseNumbers()
        );

    }

    private static void runGeneration(TestCase... testCases) throws Exception {

        for( TestCase tc : testCases ) {

            //Basic validation:
            Preconditions.checkState(tc.getTestName() != null, "Test case name is null");

            //Run through each test case:
            File testBaseDir = new File(OUTPUT_DIR, tc.getTestName());
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
            String comment = System.getProperty("user.name") + " - " + System.currentTimeMillis();
//        StringBuilder sb = new StringBuilder(comment).append("\n");
            try (OutputStream os = new BufferedOutputStream(new FileOutputStream(new File(testBaseDir, "nd4jEnvironmentInfo.json")))) {
                Enumeration<Object> e = properties.keys();
                while (e.hasMoreElements()) {
                    Object k = e.nextElement();
                    Object v = properties.get(k);
                    properties.setProperty(k.toString(), v == null ? "null" : v.toString());
                }
                properties.store(os, comment);
            }


            //First: if test is a random init test: generate the config, and save it
            MultiLayerNetwork mln = null;
            ComputationGraph cg = null;
            Model m;
            boolean isMLN;
            if (tc.getTestType() == TestCase.TestType.RANDOM_INIT) {
                Object config = tc.getConfiguration();
                String json;
                if (config instanceof MultiLayerConfiguration) {
                    MultiLayerConfiguration mlc = (MultiLayerConfiguration) config;
                    isMLN = true;
                    json = mlc.toJson();
                    mln = new MultiLayerNetwork(mlc);
                    mln.init();
                    m = mln;
                } else {
                    ComputationGraphConfiguration cgc = (ComputationGraphConfiguration) config;
                    isMLN = false;
                    json = cgc.toJson();
                    cg = new ComputationGraph(cgc);
                    cg.init();
                    m = cg;
                }

                File configFile = new File(testBaseDir, "config." + (isMLN ? "mlc.json" : "cgc.json"));
                FileUtils.writeStringToFile(configFile, json);
                log.info("RANDOM_INIT test - saved configuration: {}", configFile.getAbsolutePath());
                File savedModel = new File(testBaseDir, IntegrationTestRunner.RANDOM_INIT_UNTRAINED_MODEL_FILENAME);
                ModelSerializer.writeModel(m, savedModel, true);
                log.info("RANDOM_INIT test - saved randomly initialized model to: {}", savedModel.getAbsolutePath());
            } else {
                //Pretrained model
                m = tc.getPretrainedModel();
                isMLN = (m instanceof MultiLayerNetwork);
                if (isMLN) {
                    mln = (MultiLayerNetwork) m;
                } else {
                    cg = (ComputationGraph) m;
                }
            }


            //Generate predictions to compare against
            if (tc.isTestPredictions()) {
                List<Pair<INDArray[], INDArray[]>> inputs = tc.getPredictionsTestData();
                Preconditions.checkState(inputs != null && inputs.size() > 0, "Input data is null or length 0 for test: %s", tc.getTestName());


                File predictionsTestDir = new File(testBaseDir, "predictions");
                predictionsTestDir.mkdirs();

                int count = 0;
                if (isMLN) {
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
                } else {
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
                }

                log.info("Saved predictions for {} inputs to disk in directory: {}", tc.getTestName(), predictionsTestDir);
            }

            //Compute and save gradients:
            if (tc.isTestGradients()) {
                MultiDataSet data = tc.getGradientsTestData();
                INDArray gradientFlat;
                if (isMLN) {
                    mln.setInput(data.getFeatures(0));
                    mln.setLabels(data.getLabels(0));
                    mln.setLayerMaskArrays(data.getFeaturesMaskArray(0), data.getLabelsMaskArray(0));
                    mln.computeGradientAndScore();
                    gradientFlat = mln.getFlattenedGradients();
                } else {
                    cg.setInputs(data.getFeatures());
                    cg.setLabels(data.getLabels());
                    cg.setLayerMaskArrays(data.getFeaturesMaskArrays(), data.getLabelsMaskArrays());
                    cg.computeGradientAndScore();
                    gradientFlat = cg.getFlattenedGradients();
                }

                File gFlatFile = new File(testBaseDir, IntegrationTestRunner.FLAT_GRADIENTS_FILENAME);
                IntegrationTestRunner.write(gradientFlat, gFlatFile);

                //Also save the gradient param table:
                Map<String, INDArray> g = m.gradient().gradientForVariable();
                File gradientDir = new File(testBaseDir, "gradients");
                gradientDir.mkdir();
                for (String s : g.keySet()) {
                    File f = new File(gradientDir, s + ".bin");
                    IntegrationTestRunner.write(g.get(s), f);
                }
            }

            //Test pretraining
            if(tc.isTestUnsupervisedTraining()){
                log.info("Performing layerwise pretraining");
                MultiDataSetIterator iter = tc.getUnsupervisedTrainData();

                INDArray paramsPostTraining;
                if(isMLN){
                    int[] layersToTrain = tc.getUnsupervisedTrainLayersMLN();
                    Preconditions.checkState(layersToTrain != null, "Layer indices must not be null");
                    DataSetIterator dsi = new MultiDataSetWrapperIterator(iter);

                    for( int i : layersToTrain){
                        mln.pretrainLayer(i, dsi);
                    }
                    paramsPostTraining = mln.params();
                } else {
                    String[] layersToTrain = tc.getUnsupervisedTrainLayersCG();
                    Preconditions.checkState(layersToTrain != null, "Layer names must not be null");

                    for( String i : layersToTrain){
                        cg.pretrainLayer(i, iter);
                    }
                    paramsPostTraining = cg.params();
                }

                //Save params
                File f = new File(testBaseDir, IntegrationTestRunner.PARAMS_POST_UNSUPERVISED_FILENAME);
                IntegrationTestRunner.write(paramsPostTraining, f);
            }

            //Test training curves:
            if (tc.isTestTrainingCurves()) {
                MultiDataSetIterator trainData = tc.getTrainingData();
                CollectScoresListener l = new CollectScoresListener(1);
                m.setListeners(l);

                if (isMLN) {
                    mln.fit(trainData);
                } else {
                    cg.fit(trainData);
                }

                double[] scores = l.getListScore().toDoubleArray();
                File f = new File(testBaseDir, IntegrationTestRunner.TRAINING_CURVE_FILENAME);
                List<String> s = Arrays.stream(scores).mapToObj(String::valueOf).collect(Collectors.toList());
                FileUtils.writeStringToFile(f, String.join(",", s));

                if (tc.isTestParamsPostTraining()) {
                    File p = new File(testBaseDir, IntegrationTestRunner.PARAMS_POST_TRAIN_FILENAME);
                    IntegrationTestRunner.write(m.params(), p);
                }
            }


            if (tc.isTestEvaluation()) {
                IEvaluation[] evals = tc.getNewEvaluations();
                MultiDataSetIterator iter = tc.getEvaluationTestData();

                if (isMLN) {
                    DataSetIterator dsi = new MultiDataSetWrapperIterator(iter);
                    mln.doEvaluation(dsi, evals);
                } else {
                    cg.doEvaluation(iter, evals);
                }

                File evalDir = new File(testBaseDir, "evaluation");
                evalDir.mkdir();
                for (int i = 0; i < evals.length; i++) {
                    String json = evals[i].toJson();
                    File f = new File(evalDir, i + "." + evals[i].getClass().getSimpleName() + ".json");
                    FileUtils.writeStringToFile(f, json);
                }
            }

            //Don't need to do anything here re: overfitting
        }

        log.info("----- Completed test result generation -----");
    }
}
