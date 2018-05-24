package org.deeplearning4j.integration;


import com.google.common.collect.ImmutableSet;
import com.google.common.reflect.ClassPath;
import org.deeplearning4j.integration.util.CountingMultiDataSetIterator;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.MultiDataSetWrapperIterator;
import org.deeplearning4j.eval.*;
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
import org.nd4j.base.Preconditions;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.api.ops.impl.transforms.RelativeError;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.io.*;
import java.lang.reflect.Modifier;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

/**
 *
 * TODO: Other things we can + should check:
 * 1. Frozen layers: params don't change
 * 2. Iteration and epoch counts
 * 3. Parallel Inference
 *
 *
 */
@Slf4j
public class IntegrationTestRunner {

    public static final String RANDOM_INIT_UNTRAINED_MODEL_FILENAME = "Model_RANDOM_INIT_UNTRAINED.zip";
    public static final String FLAT_GRADIENTS_FILENAME = "flattenedGradients.bin";
    public static final String TRAINING_CURVE_FILENAME = "trainingCurve.csv";
    public static final String PARAMS_POST_TRAIN_FILENAME = "paramsPostTrain.bin";
    public static final String PARAMS_POST_UNSUPERVISED_FILENAME = "paramsPostUnsupervised.bin";

    public static final double MAX_REL_ERROR_SCORES = 1e-6;

    {
        try {
            setup();
        } catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    private static List<Class<?>> layerClasses = new ArrayList<>();
    private static List<Class<?>> preprocClasses = new ArrayList<>();
    private static List<Class<?>> graphVertexClasses = new ArrayList<>();
    private static List<Class<?>> evaluationClasses = new ArrayList<>();

    private static Map<Class<?>, Integer> layerConfClassesSeen = new HashMap<>();
    private static Map<Class<?>, Integer> preprocessorConfClassesSeen = new HashMap<>();
    private static Map<Class<?>, Integer> vertexConfClassesSeen = new HashMap<>();
    private static Map<Class<?>, Integer> evaluationClassesSeen = new HashMap<>();


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
        Preconditions.checkState(Nd4j.dataType() == DataBuffer.Type.FLOAT, "Integration tests must be run with float precision!");
        log.info("Starting test case: {}", tc.getTestName());
        long start = System.currentTimeMillis();

        File workingDir = testDir.newFolder();
        tc.initialize(workingDir);

        File testBaseDir = testDir.newFolder();
        new ClassPathResource("dl4j-integration-tests/" + tc.getTestName()).copyDirectory(testBaseDir);


        MultiLayerNetwork mln = null;
        ComputationGraph cg = null;
        Model m;
        boolean isMLN;
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
                isMLN = true;

                MultiLayerNetwork loaded = MultiLayerNetwork.load(savedModel, true);
                assertEquals("Configs not equal", loaded.getLayerWiseConfigurations(), mln.getLayerWiseConfigurations());
                assertEquals("Params not equal", loaded.params(), mln.params());
                assertEquals("Param table not equal", loaded.paramTable(), mln.paramTable());
            } else {
                ComputationGraphConfiguration cgc = (ComputationGraphConfiguration) config;
                cg = new ComputationGraph(cgc);
                m = cg;
                isMLN = false;

                ComputationGraph loaded = ComputationGraph.load(savedModel, true);
                assertEquals("Configs not equal", loaded.getConfiguration(), cg.getConfiguration());
                assertEquals("Params not equal", loaded.params(), mln.params());
                assertEquals("Param table not equal", loaded.paramTable(), mln.paramTable());
            }
        } else {
            m = tc.getPretrainedModel();
            isMLN = (m instanceof MultiLayerNetwork);
            if (isMLN) {
                mln = (MultiLayerNetwork) m;
            } else {
                cg = (ComputationGraph) m;
            }
        }

        //Collect information for test coverage
        collectCoverageInformation(m);


        //Check network output (predictions)
        if (tc.isTestPredictions()) {
            log.info("Checking predictions: saved output vs. initialized model");


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

                    //Load the previously saved array
                    File outFile = new File(predictionsTestDir, "output_" + (count++) + "_0.bin");
                    INDArray outSaved;
                    try (DataInputStream dis = new DataInputStream(new FileInputStream(outFile))) {
                        outSaved = Nd4j.read(dis);
                    }

                    assertEquals("Predictions do not match saved predictions - output " + count, outSaved, out);
                }
            } else {
                for (Pair<INDArray[], INDArray[]> p : inputs) {
                    INDArray[] out = cg.output(false, p.getFirst(), p.getSecond(), null);

                    //Save the array(s)...
                    INDArray[] outSaved = new INDArray[out.length];
                    for (int i = 0; i < out.length; i++) {
                        File outFile = new File(predictionsTestDir, "output_" + (count++) + "_" + i + ".bin");
                        try (DataInputStream dis = new DataInputStream(new FileInputStream(outFile))) {
                            outSaved[i] = Nd4j.read(dis);
                        }
                    }

                    assertArrayEquals("Predictions do not match saved predictions - output " + count, outSaved, out);
                }
            }

            checkLayerClearance(m);
        }


        //Test gradients
        if (tc.isTestGradients()) {
            log.info("Checking gradients: saved output vs. initialized model");

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
            INDArray gradientFlatSaved = read(gFlatFile);

            assertEquals("Saved flattened gradients: not equal", gradientFlatSaved, gradientFlat);

            //Load the gradient table:
            File gradientDir = new File(testBaseDir, "gradients");
            for (File f : gradientDir.listFiles()) {
                if (!f.isFile()) {
                    continue;
                }
                String key = f.getName();
                key = key.substring(0, key.length() - 4); //remove ".bin"
                INDArray loaded = read(f);
                INDArray now = m.gradient().gradientForVariable().get(key);
                assertEquals("Gradient is not equal for parameter: " + key, loaded, now);
            }
        }

        //Test layerwise pretraining
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

            File f = new File(testBaseDir, IntegrationTestRunner.PARAMS_POST_UNSUPERVISED_FILENAME);
            INDArray expParams = read(f);

            INDArray exceedsRelError = exceedsRelError(expParams, paramsPostTraining, tc.getMaxRelativeErrorPretrainParams(),
                    tc.getMinAbsErrorPretrainParams());
            int count = exceedsRelError.sumNumber().intValue();
            assertEquals("Number of parameters exceeding relative error", 0, count);

            //Set params to saved ones - to avoid accumulation of roundoff errors causing later failures...
            m.setParams(expParams);
        }


        //Test training curves:
        if (tc.isTestTrainingCurves() || tc.isTestParamsPostTraining()) {
            MultiDataSetIterator trainData = tc.getTrainingData();
            boolean isTbptt;
            int tbpttLength;
            if(isMLN){
                isTbptt = mln.getLayerWiseConfigurations().getBackpropType() == BackpropType.TruncatedBPTT;
                tbpttLength = mln.getLayerWiseConfigurations().getTbpttFwdLength();
            } else {
                isTbptt = cg.getConfiguration().getBackpropType() == BackpropType.TruncatedBPTT;
                tbpttLength = cg.getConfiguration().getTbpttFwdLength();
            }

            CountingMultiDataSetIterator countingIter = new CountingMultiDataSetIterator(trainData, isTbptt, tbpttLength);
            CollectScoresListener l = new CollectScoresListener(1);
            m.setListeners(l);

            int iterBefore;
            int epochBefore;
            int iterAfter;
            int epochAfter;

            Map<String,INDArray> frozenParamsBefore = getFrozenLayerParamCopies(m);
            if (isMLN) {
                iterBefore = mln.getIterationCount();
                epochBefore = mln.getEpochCount();
                mln.fit(countingIter);
                iterAfter = mln.getIterationCount();
                epochAfter = mln.getEpochCount();
            } else {
                iterBefore = cg.getConfiguration().getIterationCount();
                epochBefore = cg.getConfiguration().getEpochCount();
                cg.fit(countingIter);
                iterAfter = cg.getConfiguration().getIterationCount();
                epochAfter = cg.getConfiguration().getEpochCount();
            }

            //Check that frozen params (if any) haven't changed during training:
            checkFrozenParams(frozenParamsBefore, m);

            //Validate the iteration and epoch counts - both for the net, and for the layers
            int newIters = countingIter.getCurrIter();
            assertEquals(iterBefore + newIters, iterAfter);
            assertEquals(epochBefore + 1, epochAfter);
            validateLayerIterCounts(m, epochBefore + 1, iterBefore+newIters);   //TODO CURRENTLY FAILING
            double[] scores = l.getListScore().toDoubleArray();

            File f = new File(testBaseDir, IntegrationTestRunner.TRAINING_CURVE_FILENAME);
            String[] s = FileUtils.readFileToString(f).split(",");

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
                File p = new File(testBaseDir, IntegrationTestRunner.PARAMS_POST_TRAIN_FILENAME);
                INDArray paramsExp = read(p);
                INDArray z = exceedsRelError(m.params(), paramsExp, tc.getMaxRelativeErrorParamsPostTraining(), tc.getMinAbsErrorParamsPostTraining());
                int count = z.sumNumber().intValue();
                assertEquals("Number of params exceeded max relative error", 0, count);
            }

            checkLayerClearance(m);
        }

        //Check evaluation:
        if (tc.isTestEvaluation()) {
            log.info("Testing evaluation");
            IEvaluation[] evals = tc.getNewEvaluations();
            MultiDataSetIterator iter = tc.getEvaluationTestData();

            if (isMLN) {
                DataSetIterator dsi = new MultiDataSetWrapperIterator(iter);
                mln.doEvaluation(dsi, evals);
            } else {
                cg.doEvaluation(iter, evals);
            }

            File evalDir = new File(testBaseDir, "evaluation");
            for (int i = 0; i < evals.length; i++) {
                File f = new File(evalDir, i + "." + evals[i].getClass().getSimpleName() + ".json");
                String json = FileUtils.readFileToString(f);
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

                checkLayerClearance(m);
            }
        }

        //Check model serialization
        {
            log.info("Testing model serialization");

            File f = testDir.newFile();
            f.delete();

            ModelSerializer.writeModel(m, f, true);
            if (isMLN) {
                MultiLayerNetwork restored = MultiLayerNetwork.load(f, true);
                assertEquals(mln.getLayerWiseConfigurations(), restored.getLayerWiseConfigurations());
                assertEquals(mln.params(), restored.params());
            } else {
                ComputationGraph restored = ComputationGraph.load(f, true);
                assertEquals(cg.getConfiguration(), restored.getConfiguration());
                assertEquals(cg.params(), restored.params());
            }

            System.gc();
        }


        //Check parallel inference
        if (tc.isTestParallelInference()) {

            List<Pair<INDArray[], INDArray[]>> inputs = tc.getPredictionsTestData();

            int numThreads = 2; //TODO allow customization of this?

            List<INDArray[]> exp = new ArrayList<>();
            for(Pair<INDArray[], INDArray[]> p : inputs){
                INDArray[] out;
                if(isMLN){
                    INDArray fm = p.getSecond() == null ? null : p.getSecond()[0];
                    out = new INDArray[]{mln.output(p.getFirst()[0], true, fm, null)};
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
                if (isMLN) {
                    mln.fit(toOverfit);
                } else {
                    cg.fit(toOverfit);
                }
            }

            //Check:
            INDArray[] output;
            if (isMLN) {
                mln.setLayerMaskArrays(toOverfit.getFeaturesMaskArray(0), null);
                output = new INDArray[]{mln.output(toOverfit.getFeatures(0))};
            } else {
                cg.setLayerMaskArrays(toOverfit.getFeaturesMaskArrays(), null);
                output = cg.output(toOverfit.getFeatures());
            }

            for (int i = 0; i < output.length; i++) {
                INDArray z = exceedsRelError(output[i], toOverfit.getLabels(i), tc.getMaxRelativeErrorOverfit(), tc.getMinAbsErrorOverfit());
                int count = z.sumNumber().intValue();
                if (count > 0) {
                    System.out.println(output[i]);
                    System.out.println(toOverfit.getLabels(i));
                    INDArray re = relativeError(output[i], toOverfit.getLabels(i), tc.getMinAbsErrorOverfit());
                    System.out.println("Relative error:");
                    System.out.println(re);
                }
                assertEquals("Number of outputs exceeded max relative error", 0, count);
            }

            checkLayerClearance(m);
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

    public static void checkFrozenParams(Map<String,INDArray> copiesBeforeTraining, Model m){
        for(Map.Entry<String,INDArray> e : copiesBeforeTraining.entrySet()){
            INDArray actual = m.getParam(e.getKey());
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
        long numNaN1 = Nd4j.getExecutioner().exec(new MatchCondition(a1, Conditions.isNan()), Integer.MAX_VALUE).getInt(0);
        long numNaN2 = Nd4j.getExecutioner().exec(new MatchCondition(a2, Conditions.isNan()), Integer.MAX_VALUE).getInt(0);
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

}
