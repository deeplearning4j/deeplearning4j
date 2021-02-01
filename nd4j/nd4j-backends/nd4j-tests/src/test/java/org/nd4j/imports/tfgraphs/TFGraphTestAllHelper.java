<<<<<<< HEAD
/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

package org.nd4j.imports.tfgraphs;

import com.google.common.io.Files;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.math.NumberUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.nd4j.autodiff.execution.NativeGraphExecutioner;

import org.nd4j.autodiff.execution.conf.ExecutionMode;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCloseMemoryMgr;
import org.nd4j.autodiff.samediff.internal.memory.CloseValidationMemoryMgr;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.imports.tfgraphs.listener.OpExecOrderListener;
import org.nd4j.imports.listeners.ExecPrintListener;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.function.BiFunction;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.string.NDArrayStrings;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.common.resources.strumpf.ResourceFile;
import org.nd4j.common.resources.strumpf.StrumpfResolver;
import org.nd4j.samediff.frameworkimport.tensorflow.importer.TensorflowFrameworkImporter;
import org.nd4j.samediff.frameworkimport.tensorflow.ir.TensorflowIRGraph;
import org.nd4j.samediff.frameworkimport.tensorflow.ir.TensorflowIRGraphRunner;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.core.io.support.ResourcePatternResolver;
import org.tensorflow.framework.GraphDef;

import java.io.*;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static org.junit.Assert.*;
import static org.nd4j.imports.tfgraphs.TFGraphsSkipNodes.skipNode;

/**
 * Created by susaneraly on 11/6/17.
 */
@Slf4j
public class TFGraphTestAllHelper {
    public static final String resourceFolderVar = "DL4J_TEST_RESOURCES";

    public enum ExecuteWith {
        SAMEDIFF, LIBND4J, JUST_PRINT
    }

    public static class DefaultGraphLoader implements BiFunction<File,String,SameDiff> {
        @Override
        public SameDiff apply(File file, String name) {
            try {
                GraphDef graphDef = GraphDef.parseFrom(Files.toByteArray(file));
                System.out.println("Processing graph: \n" + graphDef.toString());
            } catch (IOException e) {
                e.printStackTrace();
            }

            TensorflowFrameworkImporter tensorflowFrameworkImporter = new TensorflowFrameworkImporter();
            return tensorflowFrameworkImporter.runImport(file.getAbsolutePath(),Collections.emptyMap());
        }
    }

    public static final DefaultGraphLoader LOADER = new DefaultGraphLoader();

    @BeforeClass
    public void beforeClass(){
        log.info("Starting tests for class: " + getClass().getName());
    }

    @Before
    public void setup(){
        Nd4j.setDataType(DataType.FLOAT);
    }

    @After
    public void tearDown() {
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(false);
    }

    private static ExecutorConfiguration configuration = ExecutorConfiguration.builder()
            .executionMode(ExecutionMode.SEQUENTIAL)
            .profilingMode(OpExecutioner.ProfilingMode.DISABLED)
            .gatherTimings(true)
            .outputMode(OutputMode.VARIABLE_SPACE)
            .build();

    protected static List<Object[]> fetchTestParams(String baseDir, String modelFileName, ExecuteWith executeWith, File localTestDir) throws IOException {
        String[] modelNames = modelDirNames(baseDir, executeWith, modelFileName);
        List<Object[]> modelParams = new ArrayList<>();
        for (int i = 0; i < modelNames.length; i++) {
            Object[] currentParams = new Object[4];
            currentParams[0] = inputVars(modelNames[i], baseDir, localTestDir); //input variable map - could be null
            currentParams[1] = outputVars(modelNames[i], baseDir, localTestDir); //saved off predictions
            currentParams[2] = modelNames[i];
            currentParams[3] = localTestDir;
            modelParams.add(currentParams);
        }
        return modelParams;
    }

    protected static void checkOnlyOutput(Map<String, INDArray> inputs, Map<String, INDArray> predictions, String modelName,
                                          String baseDir, String modelFilename, ExecuteWith execType, BiFunction<File,String,SameDiff> loader,
                                          Double maxRelErrorOverride, Double minAbsErrorOverride, boolean printArraysDebugging) throws IOException {
        Preconditions.checkArgument((maxRelErrorOverride == null) == (minAbsErrorOverride == null), "Both maxRelErrorOverride and minAbsErrorOverride" +
                " must be null or both must be provided");
        Nd4j.EPS_THRESHOLD = 1e-3;

        Set<String> outputsToCheck = new HashSet<>();
        for(String s : predictions.keySet()) {
            // we need to convert name from python name format with . on indices, to :. i.e.: output.1 -> output:1
            if (s.matches(".*\\.\\d+")) {
                int idx = s.lastIndexOf('.');
                s = s.substring(0, idx) + ":" + s.substring(idx + 1);
            }
            outputsToCheck.add(s);
        }

        Pair<SameDiff,Map<String,INDArray>> p = getGraphAfterExec(baseDir, modelFilename, modelName, inputs, execType, loader, null, outputsToCheck, printArraysDebugging);
        SameDiff graph = p.getFirst();
        Map<String,INDArray> sameDiffPredictions = p.getSecond();
//        SameDiff graph = graphLoaderFunction.apply(new ClassPathResource(baseDir + "/" + modelName + "/" + modelFilename).getFile(), modelName);
        //Collect coverage info about ops
       /* TensorflowFrameworkImporter tensorflowFrameworkImporter = new TensorflowFrameworkImporter();
        File oldModel = new ClassPathResource(baseDir + "/" + modelName + "/" + modelFilename).getFile();
        GraphDef g  = GraphDef.parseFrom(IOUtils.toByteArray(oldModel.toURI()));
        TensorflowIRGraph tensorflowIRGraph = new TensorflowIRGraph(g,tensorflowFrameworkImporter.getOpDefList(),tensorflowFrameworkImporter.getRegistry());
        TensorflowIRGraphRunner tensorflowIRGraphRunner = new TensorflowIRGraphRunner(tensorflowIRGraph,p.getFirst().inputs(),outputsToCheck.stream().collect(Collectors.toList()));
        Map<String,INDArray> outputs = tensorflowIRGraphRunner.run(inputs);
        SameDiff oldForComparison = TFGraphMapper.importGraph(oldModel);
        Map<String,INDArray> oldOutputs = oldForComparison.outputAll(inputs);
        */
        OpValidation.collectTensorflowImportCoverage(graph);

        if (!execType.equals(ExecuteWith.JUST_PRINT)) {
            assertTrue("No predictions to validate", predictions.keySet().size() > 0);
            for (String outputNode : predictions.keySet()) {
                INDArray nd4jPred = null;
                INDArray tfPred = null;

                String nd4jNode = outputNode;

                // we need to convert name from python name format with . on indices, to :. i.e.: output.1 -> output:1
                if (outputNode.contains("."))
                    nd4jNode = outputNode.replaceAll("\\.", ":");

                try {
                    nd4jPred = sameDiffPredictions.get(nd4jNode);
                } catch (NullPointerException e) {
                    throw new NullPointerException("Can't find SameDiff variable with name [" + nd4jNode + "]");
                }

                try {
                    tfPred = predictions.get(outputNode);
                } catch (NullPointerException e) {
                    throw new NullPointerException("Can't find predicted variable with name [" + outputNode + "]");
                }

                assertNotNull(nd4jPred);
                assertNotNull(tfPred);

                if(maxRelErrorOverride == null) {
                    long[] sTf = tfPred.shape();
                    long[] sNd4j = nd4jPred.shape();
                    assertArrayEquals("Shapes for node \"" + outputNode + "\" are not equal: TF: " + Arrays.toString(sTf) + " vs SD: " + Arrays.toString(sNd4j), sTf, sNd4j);

                    // TODO: once we add more dtypes files - this should be removed
                    if (tfPred.dataType() != nd4jPred.dataType())
                        nd4jPred = nd4jPred.castTo(tfPred.dataType());

                    boolean eq = getEqualityFunction(modelName, outputNode, tfPred, nd4jPred).apply(tfPred, nd4jPred);

                    if(!eq){
                        //Check for both NaN, both inf
                        if(tfPred.dataType().isFPType() && tfPred.equalShapes(nd4jPred) && tfPred.isNaN().castTo(DataType.INT).sumNumber().intValue() == tfPred.length()
                                && nd4jPred.isNaN().castTo(DataType.INT).sumNumber().intValue() == nd4jPred.length()){
                            //All NaNs in both arrays
                            eq = true;
                        } else if(tfPred.dataType().isFPType() && tfPred.equalShapes(nd4jPred) && tfPred.isInfinite().castTo(DataType.INT).sumNumber().intValue() == tfPred.length()
                                && nd4jPred.isInfinite().castTo(DataType.INT).sumNumber().intValue() == nd4jPred.length()){
                            //All infinite in both arrays. But need to check that it's all positive vs. negative infinite in both cases...
                            NdIndexIterator iter = new NdIndexIterator(tfPred.shape());
                            eq = true;
                            while(iter.hasNext()) {
                                long[] next = iter.next();
                                //Already know they are both infinite, only question is whether they are both positive and negative
                                double d1 = tfPred.getDouble(next);
                                double d2 = nd4jPred.getDouble(next);
                                if((d1 > 0) != (d2 > 0)){
                                    eq = false;
                                    break;
                                }
                            }
                        }

                        if(!eq) {
                            NDArrayStrings s = new NDArrayStrings();
                            String s1 = s.format(tfPred, false);
                            String s2 = s.format(nd4jPred, false);
                            System.out.print("TF: ");
                            System.out.println(tfPred.toStringFull());
                            System.out.print("SD: ");
                            System.out.println(nd4jPred.toStringFull());
                        }
                    }

                    assertTrue("Predictions do not match on " + modelName + ", node " + outputNode, eq);
                } else {

                    if(!tfPred.equalShapes(nd4jPred)) {
                        fail("Output node \"" + outputNode + "\" SameDiff output shape does not match TF output shape: SameDiff shape: " +
                                Arrays.toString(nd4jPred.shape()) + " vs. TF shape: " + Arrays.toString(tfPred.shape()));
                    }

                    if(tfPred.dataType() != nd4jPred.dataType()) {
                        fail("Output node \"" + outputNode + "\" SameDiff output datatype does not match TF output : SameDiff type: " +
                                nd4jPred.dataType() + " vs. TF datatype: " + tfPred.dataType());
                    }

                    if(!tfPred.dataType().isFPType()){
                        //Can't do relative error on long type...
                        tfPred = tfPred.castTo(DataType.DOUBLE);
                        nd4jPred = nd4jPred.castTo(DataType.DOUBLE);
                    }

                    INDArray diff = Transforms.abs(tfPred.sub(nd4jPred), false);
                    INDArray absErrorMask = diff.gte(minAbsErrorOverride).castTo(tfPred.dataType());   //value 1 if x[i] > minAbsError; value 0 otherwise. Used to get rid of 1e-30 vs. 1e-29 type failures
                    INDArray sumAbs = Transforms.abs(tfPred, true).addi(Transforms.abs(nd4jPred, true));
                    BooleanIndexing.replaceWhere(sumAbs, 1.0, Conditions.equals(0.0));  //Can only get 0.0 if both are zeros - need to avoid 0/0=NaN
                    INDArray relError = diff.divi(sumAbs);
                    relError.muli(absErrorMask);


                    /*
                    Try to detect bad test.
                    The idea: suppose all values are small, and are excluded due to minAbsError threshold
                    i.e., all 1e-5 vs. -1e-5 with min abs error of 1e-4
                    */
                    //TODO FIX ME
                    INDArray maxAbs = Transforms.max(Transforms.abs(tfPred.castTo(DataType.DOUBLE), true), Transforms.abs(nd4jPred.castTo(DataType.DOUBLE), true), true);
                    long countMaxAbsGTThreshold = maxAbs.gte(minAbsErrorOverride).castTo(DataType.INT).sumNumber().intValue();
                    long countNotMasked = absErrorMask.sumNumber().intValue();  //Values are 0 or 1... if all 0s -> nothing being tested
                    if(countNotMasked == 0 && countMaxAbsGTThreshold == 0){
                        fail("All values for node " + outputNode + " are masked out due to minAbsError=" + minAbsErrorOverride +
                                " and max values are all less than minAbsError - nothing can be tested here");
                    }

                    int countExceeds = Nd4j.getExecutioner().exec(new MatchCondition(relError, Conditions.greaterThan(maxRelErrorOverride))).getInt(0);

                    double maxRE = -1;
                    if(countExceeds > 0){
                        maxRE = relError.maxNumber().doubleValue();
                    }


                    assertEquals( outputNode + ": " + countExceeds + " values exceed maxRelError=" + maxRelErrorOverride
                            + " with minAbsError=" + minAbsErrorOverride + "; largest observed relError=" + maxRE, 0, countExceeds);
                }
            }
            log.info("TEST {} PASSED with {} arrays compared...", modelName, predictions.keySet().size());
        }

        //Serialize and deserialize, check equality:
        ByteBuffer serialized = graph.asFlatBuffers(true);
        Preconditions.checkNotNull(serialized, "Serialization failed? Null output");
        OpValidation.checkDeserializedEquality(graph, serialized, new TestCase(graph).testName(modelName).placeholderValues(inputs));


        Nd4j.EPS_THRESHOLD = 1e-5;
    }

    public static void checkIntermediate(Map<String, INDArray> inputs, String modelName, String baseDir, String modelFileName,
                                         ExecuteWith execType, File localTestDir, boolean printArraysDebugging) throws IOException {
        checkIntermediate(inputs, modelName, baseDir, modelFileName, execType, LOADER, null, null, localTestDir, printArraysDebugging);
    }

    public static void checkIntermediate(Map<String, INDArray> inputs, String modelName, String baseDir, String modelFileName,
                                         ExecuteWith execType, BiFunction<File,String,SameDiff> loader,
                                         Double maxRelErrorOverride, Double minAbsErrorOverride, File localTestDir, boolean printArraysDebugging) throws IOException {
        Preconditions.checkArgument((maxRelErrorOverride == null) == (minAbsErrorOverride == null), "Both maxRelErrorOverride and minAbsErrorOverride" +
                " must be null or both must be provided");
        Nd4j.EPS_THRESHOLD = 1e-3;
        OpExecOrderListener listener = new OpExecOrderListener();       //Used to collect exec order
        Pair<SameDiff, Map<String,INDArray>> p = getGraphAfterExec(baseDir, modelFileName, modelName, inputs, execType, loader, Collections.singletonList(listener), null, printArraysDebugging);
        SameDiff graph = p.getFirst();
        Map<String,INDArray> sdPredictions = p.getSecond();

        //Collect coverage info about ops
        OpValidation.collectTensorflowImportCoverage(graph);

        if (!execType.equals(ExecuteWith.JUST_PRINT)) {
            int count = 0;
            //Evaluate the nodes in their execution order - this is useful for debugging (as we want the *first* failure
            // to be detected before later failures)
            List<String> varNames = new ArrayList<>();
            Map<String,SameDiffOp> fns = graph.getOps();
            List<String> execOrder = listener.getOpNamesList();
            for(String opName : execOrder){
                String[] outputs = graph.getOutputsForOp(fns.get(opName).getOp());
                Collections.addAll(varNames, outputs);
            }

            for (String varName : varNames) {
                if (!inputs.containsKey(varName)) { //avoiding placeholders
                    INDArray tfValue = intermediateVars(modelName, baseDir, varName, localTestDir);
                    if (tfValue == null) {
                        continue;
                    }
                    log.info("Starting check: variable {}", varName);
                    if (skipNode(modelName, varName)) {
                        log.info("\n\tFORCING no check on " + varName);
                    } else {
                        //assertArrayEquals("Shape not equal on node " + varName, tfValue.shape(), graph.getVariable(varName).getShape());
                        INDArray sdVal = sdPredictions.get(varName);
                        if(maxRelErrorOverride != null){
                            INDArray diff = Transforms.abs(tfValue.sub(sdVal), false);
                            INDArray absErrorMask = diff.gte(minAbsErrorOverride);   //value 1 if x[i] > minAbsError; value 0 otherwise. Used to get rid of 1e-30 vs. 1e-29 type failures
                            INDArray sumAbs = Transforms.abs(tfValue, true).addi(Transforms.abs(sdVal, true));
                            BooleanIndexing.replaceWhere(sumAbs, 1.0, Conditions.equals(0.0));  //Can only get 0.0 if both are zeros - need to avoid 0/0=NaN
                            INDArray relError = diff.divi(sumAbs);
                            relError.muli(absErrorMask);

                            int countExceeds = Nd4j.getExecutioner().exec(new MatchCondition(relError, Conditions.greaterThan(maxRelErrorOverride))).getInt(0);

                            double maxRE = -1;
                            //Mainly used for analysis in debugger:
                            DifferentialFunction op = null;
                            String[] opInputs = null;
                            if(countExceeds > 0){
                                maxRE = relError.maxNumber().doubleValue();
                                //Find the op that this variable is produced by
                                op = graph.getVariableOutputOp(varName);
                                opInputs = graph.getInputsForOp(op);
                            }


                            assertEquals( varName + ": " + countExceeds + " values exceed maxRelError=" + maxRelErrorOverride
                                    + " with minAbsError=" + minAbsErrorOverride + "; largest observed relError=" + maxRE, 0, countExceeds);
                        } else {
//                            assertEquals("Value not equal on node " + varName, tfValue, sdVal);
                            if(tfValue.equals(sdVal)){
                                System.out.println("Pass: " + varName);
                            } else {
                                System.out.println("FAIL: " + varName);
                                System.out.println("TF:\n" + tfValue);
                                System.out.println("SD:\n" + sdVal);
                            }

                        }
                        log.info("Values and shapes equal for {}", varName);
                        count++;
                    }

                }
            }

            assertTrue("No intermediate variables were checked", count > 0);
        }

        Nd4j.EPS_THRESHOLD = 1e-5;
    }

    public static Pair<SameDiff, Map<String,INDArray>> getGraphAfterExec(String baseDir, String modelFilename, String modelName, Map<String, INDArray> inputs,
                                                                         ExecuteWith executeWith, BiFunction<File,String,SameDiff> graphLoaderFunction, List<Listener> listeners,
                                                                         Set<String> requiredOutputs, boolean printArraysDebugging) throws IOException {
        log.info("RUNNING TEST {}...", modelName);
        SameDiff graph = graphLoaderFunction.apply(new ClassPathResource(baseDir + "/" + modelName + "/" + modelFilename).getFile(), modelName);
        if(listeners != null){
            graph.setListeners(listeners);
        }

        if(printArraysDebugging) {
            graph.addListeners(new ExecPrintListener());
        }

        if(requiredOutputs == null){
            requiredOutputs = graph.variableMap().keySet();
        }

        Map<String,INDArray> outMap = null;
        if (executeWith.equals(ExecuteWith.SAMEDIFF)) {
            //Set memory manager - check that all arrays (other than the ones we requested as output)
            CloseValidationMemoryMgr mmgr = new CloseValidationMemoryMgr(graph, new ArrayCloseMemoryMgr());
           /* long tid = Thread.currentThread().getId();
            if(!graph.getSessions().containsKey(tid))
                graph.getSessions().put(tid, new InferenceSession(graph));*/
            //Execute
           // graph.getSessions().get(tid).setMmgr(mmgr);
            Map<String,String> shapes = new HashMap<>();
            inputs.entrySet().stream().forEach(entry -> {
                shapes.put(entry.getKey(),Arrays.toString(entry.getValue().shape()));
            });

            log.info("Testing inputs with names " + inputs.keySet() + " and shapes " + shapes);

            outMap = graph.output(inputs, new ArrayList<>(requiredOutputs));

            //Check that all arrays were released
            //mmgr.assertAllReleasedExcept(outMap.values());
            graph.getSessions().clear();
        } else if (executeWith.equals(ExecuteWith.LIBND4J)) {
            for (String input : inputs.keySet()) {
                graph.associateArrayWithVariable(inputs.get(input), graph.variableMap().get(input));
            }

//            val string = graph.asFlatPrint();
//            log.info("Graph structure: \n{}", string);
            val executioner = new NativeGraphExecutioner();
            val results = executioner.executeGraph(graph, configuration);

        } else if (executeWith.equals(ExecuteWith.JUST_PRINT)) {
            for (String input : inputs.keySet()) {
                graph.associateArrayWithVariable(inputs.get(input), graph.variableMap().get(input));
            }

            val string = graph.asFlatPrint();
            log.info("Graph structure: \n{}", string);
        }

        return new Pair<>(graph, outMap);
    }

    private static String[] modelDirNames(String base_dir, ExecuteWith executeWith, String modelFileName) throws IOException {
        ResourcePatternResolver resolver = new PathMatchingResourcePatternResolver(new ClassPathResource(base_dir).getClassLoader());
        Resource[] resources = resolver.getResources("classpath*:" + base_dir + "/**/" + modelFileName );
        String[] exampleNames = new String[resources.length];
        for (int i = 0; i < resources.length; i++) {
            String nestedName = resources[i].getURL().toString().split(base_dir + "/")[1];
            exampleNames[i] = nestedName.replaceAll(Pattern.quote(base_dir), "").replaceAll("/" + modelFileName, "");
        }
        return exampleNames;
    }

    protected static Map<String, INDArray> inputVars(String modelName, String base_dir, File localTestDir) throws IOException {
        return readVars(modelName, base_dir, "**.placeholder", true, localTestDir);
    }


    protected static Map<String, INDArray> outputVars(String modelName, String base_dir, File localTestDir) throws IOException {
        return readVars(modelName, base_dir, "**.prediction", true, localTestDir);
    }

    protected static Map<String, INDArray> inbetweenVars(String modelName, String base_dir, File localTestDir) throws IOException {
        return readVars(modelName, base_dir, "**.prediction_inbw", true, localTestDir);
    }


    //return readVars(modelName, base_dir, "**.prediction_inbw", true);

    /**
     * Possible for a single node to give multiple outputs
     *
     * How is a node that has a list of outputs like in the case of "node_multiple_out" work
     * Below is hardcoded for a single node
     */
    protected static INDArray intermediateVars(String modelName, String base_dir, String varName, File localTestDir) throws IOException {
        //convert varName to convention used in naming files
        // "/" replaced by "____"; followed by a digit indicating the output number followed by prediction_inbw.(shape|csv)
        if (varName.contains(":")) {
            varName = varName.replace(':', '.');
        } else {
            varName = varName + ".0";
        }
        String name = varName.replaceAll("/", "____") + ".prediction_inbw";
        Map<String, INDArray> nodeSepOutput = readVars(modelName, base_dir, name, true, localTestDir);

        boolean importNameWorkaround = false;
        if(nodeSepOutput.isEmpty()){
            //Edge case: intermediates were generated with help of import_graph_def method, which by default adds "import/" to names
            // for some reason. https://www.tensorflow.org/api_docs/python/tf/graph_util/import_graph_def
            //So many of earlier intermediate nodes test data were generated with filenames like "import___X..." instead of "X..."
            name = "import____" + name;
            nodeSepOutput = readVars(modelName, base_dir, name, true, localTestDir);
            importNameWorkaround = true;
        }

        //required check for pattern matching as there are scopes and "*" above is a greedy match
        Set<String> removeList = confirmPatternMatch(nodeSepOutput.keySet(), importNameWorkaround ? "import/" + varName : varName);
        for (String toRemove : removeList) {
            nodeSepOutput.remove(toRemove);
        }
        if(importNameWorkaround){
            return nodeSepOutput.get("import/" + varName); //this *should* return a list of the indarrays for each node
        } else {
            return nodeSepOutput.get(varName); //this *should* return a list of the indarrays for each node
        }
    }

    public static Set<String> confirmPatternMatch(Set<String> setOfNames, String varName) {
        Set<String> removeList = new HashSet<>();
        for (String name : setOfNames) {
            if (name.equals(varName)) continue;
            String[] splitByPeriod = name.split("\\.");
            //not a number - maybe another variable deeper in the same scope
            if (!NumberUtils.isNumber(splitByPeriod[splitByPeriod.length - 1])) {
                removeList.add(name);
            } else if (!String.join(".", Arrays.copyOfRange(splitByPeriod, 0, splitByPeriod.length - 1)).equals(varName)) {
                removeList.add(name);
            }
        }
        return removeList;
    }


    protected static Map<String, INDArray> readVars(String modelName, String base_dir, String pattern, boolean recursive, File localTestDir) throws IOException {
        Map<String, INDArray> varMap = new HashMap<>();
        String modelDir = base_dir + "/" + modelName;

        // key is variable name, value is data type
        val dtypes = new HashMap<String, DataType>();

        List<Pair<Resource,Resource>> resources = new ArrayList<>();
        if(recursive){
            String nameRegex = pattern.replace("**.",".*\\.") + "\\.shape";
//            File baseDir = new File(System.getProperty("java.io.tmpdir"), UUID.randomUUID().toString() + "/" + modelName);
//            baseDir.mkdirs();
//            baseDir.deleteOnExit();
//            new ClassPathResource(modelDir).copyDirectory(baseDir);

            // checking out, if local folder declared
            String localPath = System.getenv(TFGraphTestAllHelper.resourceFolderVar);
            if(localPath != null && (!localPath.contains("src/main/resources") && !localPath.contains("src\\main\\resources"))){
                localPath = FilenameUtils.concat(localPath, "src/main/resources");
            }

            // baseDir will differ, depending on run mode
            File baseDir = localPath == null ? new File(localTestDir, "extracted/" + modelName) : new File(localPath, base_dir + "/" + modelName);
            String[] arr = baseDir.list();

            if(!baseDir.exists() || arr == null || arr.length == 0){
                // we're skipping extraction if we're using local copy of dl4j-tests-resources
                if (localPath == null) {
                    baseDir.mkdirs();
                    baseDir.deleteOnExit();
                    String md = modelDir;
                    if(!md.endsWith("/") && !md.endsWith("\\")){
                        md = md + "/";
                    }
                    new ClassPathResource(md).copyDirectory(baseDir);
                } else{
                    throw new IllegalStateException("local directory declared but could not find files: " + baseDir.getAbsolutePath());
                }

            }

            LinkedList<File> queue = new LinkedList<>();
            queue.add(baseDir);

            while(!queue.isEmpty()){
                File subdir = queue.remove();
                File[] files = subdir.listFiles();
                if (files != null) {
                    for (File f : files) {
                        if (f.isDirectory()) {
                            queue.add(f);
                        } else {
                            String filename = f.getName();
                            if(filename.matches(nameRegex)){
                                File csvFile = new File(f.getAbsolutePath().replace(".shape",".csv"));
                                resources.add(new Pair<>(new FileSystemResource(f), new FileSystemResource(csvFile)));
                            } else if (filename.equals("dtypes")) {
                                List<String> stringList;

                                try (val is = new BufferedInputStream(new FileInputStream(f))) {
                                    stringList = IOUtils.readLines(is, StandardCharsets.UTF_8);

                                    for (val s:stringList) {
                                        val split = s.split("\\ ");

                                        val okey = split[0].replaceAll("____", "/");
                                        // adopt / in names
                                        val key = modelDir + "/" + okey;

                                        // parse type directly
                                        DataType value = ArrayOptionsHelper.dataType(split[1]);

                                        // adding key directly
                                        //if (dtypes.containsKey(key))
                                        //    throw new ND4JIllegalStateException("Specified key already exist: [" + key + "]");
                                        //else

                                        dtypes.put(key, value);

                                        // adding zero output duplicate (if it doesn't exist)
                                        if (key.endsWith(".0")) {
                                            val nkey = key.replaceAll("\\.0$","");
                                            if (!dtypes.containsKey(nkey)) {
                                                dtypes.put(nkey, value);
                                            }
                                        } else if (key.endsWith(":0")) {
                                            val nkey = key.replaceAll(":0$","");
                                            if (!dtypes.containsKey(nkey)) {
                                                dtypes.put(nkey, value);
                                            }
                                        }
                                    }
                                } catch (FileNotFoundException e) {
                                    stringList = new ArrayList<>();
                                }
                            }
                        }
                    }
                }
            }
        } else {
            ResourcePatternResolver resolver = new PathMatchingResourcePatternResolver(new ClassPathResource(modelDir).getClassLoader());
            Resource[] r = resolver.getResources("classpath*:" + modelDir + "/" + pattern + ".shape");
            for(Resource res : r){
                String fileName = res.getFilename();
                String varPath = modelDir + "/" + fileName;
                Resource r2 = new org.springframework.core.io.ClassPathResource(varPath.replace(".shape", ".csv"));
                resources.add(new Pair<>(res, r2));
            }

        }

//        Preconditions.checkState(!dtypes.isEmpty(), "No datatypes file was found");

        val dtype = Nd4j.dataType();
        for (int i = 0; i < resources.size(); i++) {
            URI u = resources.get(i).getFirst().getURI();
            String varName = u.toString();
            int idx = varName.indexOf(modelName);
            varName = varName.substring(idx + modelName.length()+1);    //+1 for "/"
            varName = varName.replaceAll("____","/");
            varName = varName.replaceAll(".placeholder.shape","");
            varName = varName.replaceAll(".prediction.shape","");
            varName = varName.replaceAll(".prediction_inbw.shape","");

            DataType type = dtypes.get(modelDir + "/" + varName);

            List<String> lines; //= FileUtils.readLines(new ClassPathResource(varPath).getFile(), Charset.forName("UTF-8"));
            try(InputStream is = new BufferedInputStream(resources.get(i).getFirst().getInputStream())){
                lines = IOUtils.readLines(is, StandardCharsets.UTF_8);
            }
            List<String> filtered = new ArrayList<>(lines.size());
            for(String s : lines){
                String trimmed = s.trim();
                if(!trimmed.isEmpty()){
                    filtered.add(trimmed);
                }
            }

            if(type == null){
                log.warn("DATATYPE NOT AVAILABLE FOR: {} - {}", modelName, varName);
                //Soon: this will be an exception
                type = DataType.FLOAT;
            }

            INDArray varValue;
            if(filtered.size() == 0){
                //Scalar
                String content = IOUtils.toString(resources.get(i).getSecond().getInputStream(), StandardCharsets.UTF_8);
                switch (type){
                    case DOUBLE:
                    case FLOAT:
                    case HALF:
                    case BFLOAT16:
                        varValue = Nd4j.scalar(type, parseDouble(content));
                        break;
                    case LONG:
                    case INT:
                    case SHORT:
                    case UBYTE:
                    case BYTE:
                    case UINT16:
                    case UINT32:
                    case UINT64:
                        varValue = Nd4j.scalar(type, parseLong(content));
                        break;
                    case BOOL:
                        varValue = Nd4j.scalar(parseBoolean(content));
                        break;
                    case UTF8:
                        varValue = Nd4j.scalar(content);
                        break;
                    case COMPRESSED:
                    case UNKNOWN:
                    default:
                        throw new UnsupportedOperationException("Unknown / not implemented datatype: " + type);
                }
            } else {
                int[] varShape = new int[filtered.size()];
                for( int j=0; j<filtered.size(); j++ ){
                    varShape[j] = Integer.parseInt(filtered.get(j));
                }

                try {
                    String content;
                    Pair<Resource,Resource> p = resources.get(i);
                    boolean isRef = p.getSecond().isFile() && !p.getSecond().exists();

                    InputStream stream;
                    if(isRef){
                        //Slight hack for loading strumpf reference files
                        File r = new StrumpfResolver().localCacheRoot();
                        String path = p.getSecond().getFile() + StrumpfResolver.REF;
                        File f = ResourceFile.fromFile(path).localFile(r);
                        stream = new BufferedInputStream(new FileInputStream(f));
                    } else {
                        stream = new BufferedInputStream(resources.get(i).getSecond().getInputStream());
                    }

                    try(InputStream is = stream){
                        content = String.join("\n", IOUtils.readLines(is, StandardCharsets.UTF_8));
                    }

                    if (content.isEmpty()) {
                        //Should be zeros in shape
                        boolean foundZero = false;
                        for( int s : varShape){
                            foundZero |= (s == 0);
                        }
                        if(foundZero){
                            varValue = Nd4j.create(type, ArrayUtil.toLongArray(varShape));
                        } else {
                            throw new IllegalStateException("Empty data but non-empty shape: " + resources.get(i).getSecond());
                        }
                    } else {
                        if(varShape.length == 1 && varShape[0] == 0)        //Annoyingly, some scalars have shape [0] instead of []
                            varShape = new int[0];

                        String[] cLines = content.split("\n");
                        switch (type){
                            case DOUBLE:
                            case FLOAT:
                            case HALF:
                            case BFLOAT16:
                                double[] dArr = new double[cLines.length];
                                int x=0;
                                while(x < dArr.length){
                                    dArr[x] = parseDouble(cLines[x]);
                                    x++;
                                }
                                varValue = Nd4j.createFromArray(dArr).castTo(type).reshape('c', varShape);
                                break;
                            case LONG:
                            case INT:
                            case SHORT:
                            case UBYTE:
                            case BYTE:
                            case UINT16:
                            case UINT32:
                            case UINT64:
                                long[] lArr = new long[cLines.length];
                                int y=0;
                                while(y < lArr.length){
                                    lArr[y] = parseLong(cLines[y]);
                                    y++;
                                }
                                varValue = Nd4j.createFromArray(lArr).castTo(type).reshape('c', varShape);
                                break;
                            case BOOL:
                                boolean[] bArr = new boolean[cLines.length];
                                int z=0;
                                while(z < bArr.length){
                                    bArr[z] = parseBoolean(cLines[z]);
                                    z++;
                                }
                                varValue = Nd4j.createFromArray(bArr).reshape('c', varShape);
                                break;
                            case UTF8:
                                varValue = Nd4j.create(cLines).reshape('c', varShape);
                                break;
                            case COMPRESSED:
                            case UNKNOWN:
                            default:
                                throw new UnsupportedOperationException("Unknown / not implemented datatype: " + type);
                        }
                    }
                } catch (NumberFormatException e) {
                    log.warn("Error parsing number", e);
                    continue;
                }
            }

            varMap.put(varName, varValue);
        }
        return varMap;
    }

    private static long parseLong(String line){
        line = line.trim();       //Handle whitespace
        if(line.matches("-?\\d+\\.0+")){
            //Annoyingly, some integer data is stored with redundant/unnecessary zeros - like "-7.0000000"
            return Long.parseLong(line.substring(0, line.indexOf('.')));
        } else {
            return Long.parseLong(line);
        }
    }

    private static double parseDouble(String line){
        line = line.trim();   //Handle whitespace - some lines are like "      -inf"
        if("nan".equalsIgnoreCase(line)){
            return Double.NaN;
        } else if("inf".equalsIgnoreCase(line)) {
            return Double.POSITIVE_INFINITY;
        } else if("-inf".equalsIgnoreCase(line)){
            return Double.NEGATIVE_INFINITY;
        } else {
            return Double.parseDouble(line);
        }
    }

    private static boolean parseBoolean(String line){
        line = line.trim();
        if(line.matches("1(\\.0*)?")){          //Booleans are ocassionally represented like 1.000000 or 0.000000
            return true;
        } else if(line.matches("0(\\.0*)?")){
            return false;
        }
        return Boolean.parseBoolean(line);
    }


    public static Pair<Double,Double> testPrecisionOverride(String testName){
        if("conv_4".equalsIgnoreCase(testName)) {
            //Most values: around 1k. So this is the 6th significant figure, which is OK
            return new Pair<>(1e-3, 1e-5);
        }
        return null;
    }

    public static boolean equalsWithEps(double a, double b){
        return Math.abs(a - b) <= 0.00001;
    }

    public static BiFunction<INDArray, INDArray, Boolean> getEqualityFunction(String modelName, String varName, INDArray tf, INDArray sd){
        if(modelName.startsWith("topk")){
            return (t, s) -> Nd4j.sort(t, true).equals(Nd4j.sort(s, true));
        }

        if(modelName.startsWith("empty")){
            return (t, s) -> {
                boolean areEqualShapes = t.equalShapes(s);
                boolean areEqualDataTypes = t.dataType() == s.dataType();
                return areEqualShapes && areEqualDataTypes;
            };        }

        // sum of all elements along dimesions before and after shuffle has to be the same
        if(modelName.startsWith("random_shuffle")){
            return (t, s) -> Nd4j.sort(t, true).equals(Nd4j.sort(s, true));
        }

        if(modelName.startsWith("random_normal")){
            return (t, s) -> {
                boolean areEqualShapes = t.equalShapes(s);
                double meanS = s.meanNumber().doubleValue();
                double meanT = t.meanNumber().doubleValue();
                double stdS = s.stdNumber().doubleValue();
                double stdT = t.stdNumber().doubleValue();
                double eps = 1;
                return areEqualShapes && (Math.abs(meanS-meanT) < eps) && (Math.abs(stdS-stdT) < eps);
            };        }

        if(modelName.startsWith("random_gamma")){
            return (t, s) -> {
                boolean areEqualShapes = t.equalShapes(s);
                boolean nonNegativeValues = (t.minNumber().doubleValue() > 0) && (t.minNumber().doubleValue() > 0);
                double meanS = s.meanNumber().doubleValue();
                double meanT = t.meanNumber().doubleValue();
                double stdS = s.stdNumber().doubleValue();
                double stdT = t.stdNumber().doubleValue();
                double eps = 1;
                return areEqualShapes && nonNegativeValues && (Math.abs(meanS-meanT) < eps) && (Math.abs(stdS-stdT) < eps);
            };
        }

        if(modelName.startsWith("random_poisson") || modelName.startsWith("random_poisson_v2")){
            return (t, s) -> {
                boolean areEqualShapes = t.equalShapes(s);
                boolean nonNegativeValues = (t.minNumber().doubleValue() >= 0) && (t.minNumber().doubleValue() >= 0);
                double meanS = s.meanNumber().doubleValue();
                double meanT = t.meanNumber().doubleValue();
                double stdS = s.stdNumber().doubleValue();
                double stdT = t.stdNumber().doubleValue();
                double eps = 1;
                return areEqualShapes && nonNegativeValues && (Math.abs(meanS-meanT) < eps) && (Math.abs(stdS-stdT) < eps);
            };
        }

        if(modelName.startsWith("random_uniform")|| modelName.startsWith("random_uniform_int")){
            return (t, s) -> {
                boolean areEqualShapes = t.equalShapes(s);
                double meanS = s.meanNumber().doubleValue();
                double meanT = t.meanNumber().doubleValue();
                double stdS = s.stdNumber().doubleValue();
                double stdT = t.stdNumber().doubleValue();
                double eps = 1;
                return areEqualShapes && (Math.abs(stdS-stdT) < eps) && (Math.abs(meanS-meanT) < eps);
            };
        }

        if(modelName.startsWith("alpha_dropout") || modelName.startsWith("layers_dropout") || modelName.startsWith("dropout"))
            //We can't compare dropout using simple equality due to randomness
            return (t, s) -> {
                double[] tfNums = t.ravel().toDoubleVector();
                double[] sdNums = s.ravel().toDoubleVector();

                Double seen1 = null, seen2 = null;
                for(int i = 0 ; i < tfNums.length ; i++){
                    if(!equalsWithEps(tfNums[i], sdNums[i])){

                        // if we have only seen one inequality so far, figure out which is the dropout
                        if(seen1 != null && seen2 != null){
                            if(equalsWithEps(tfNums[i], seen1) || equalsWithEps(sdNums[i], seen1)) // the dropout is in seen1
                                seen2 = null;
                            else if(equalsWithEps(tfNums[i], seen2) || equalsWithEps(sdNums[i], seen2)){ // the dropout is in seen2
                                seen1 = seen2;
                                seen2 = null;
                            } else // neither match
                                return false;
                        }

                        if(seen1 != null){
                            if(!equalsWithEps(tfNums[i], seen1) && !equalsWithEps(sdNums[i], seen1))
                                return false;
                        } else {
                            seen1 = tfNums[i];
                            seen2 = sdNums[i];
                        }
                    }
                }

                return true;
            };

        return Object::equals;
    }

}
=======
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

package org.nd4j.imports.tfgraphs;

import com.google.common.io.Files;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.math.NumberUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.nd4j.autodiff.execution.NativeGraphExecutioner;

import org.nd4j.autodiff.execution.conf.ExecutionMode;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCloseMemoryMgr;
import org.nd4j.autodiff.samediff.internal.memory.CloseValidationMemoryMgr;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.imports.tfgraphs.listener.OpExecOrderListener;
import org.nd4j.imports.listeners.ExecPrintListener;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.function.BiFunction;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.string.NDArrayStrings;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.common.resources.strumpf.ResourceFile;
import org.nd4j.common.resources.strumpf.StrumpfResolver;
import org.nd4j.samediff.frameworkimport.tensorflow.importer.TensorflowFrameworkImporter;
import org.nd4j.samediff.frameworkimport.tensorflow.ir.TensorflowIRGraph;
import org.nd4j.samediff.frameworkimport.tensorflow.ir.TensorflowIRGraphRunner;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.core.io.support.ResourcePatternResolver;
import org.tensorflow.framework.GraphDef;

import java.io.*;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static org.junit.Assert.*;
import static org.nd4j.imports.tfgraphs.TFGraphsSkipNodes.skipNode;

/**
 * Created by susaneraly on 11/6/17.
 */
@Slf4j
public class TFGraphTestAllHelper {
    public static final String resourceFolderVar = "DL4J_TEST_RESOURCES";

    public enum ExecuteWith {
        SAMEDIFF, LIBND4J, JUST_PRINT
    }

    public static class DefaultGraphLoader implements BiFunction<File,String,SameDiff> {
        @Override
        public SameDiff apply(File file, String name) {
            try {
                GraphDef graphDef = GraphDef.parseFrom(Files.toByteArray(file));
                System.out.println("Processing graph: \n" + graphDef.toString());
            } catch (IOException e) {
                e.printStackTrace();
            }

            TensorflowFrameworkImporter tensorflowFrameworkImporter = new TensorflowFrameworkImporter();
            return tensorflowFrameworkImporter.runImport(file.getAbsolutePath(),Collections.emptyMap());
        }
    }

    public static final DefaultGraphLoader LOADER = new DefaultGraphLoader();

    @BeforeClass
    public void beforeClass(){
        log.info("Starting tests for class: " + getClass().getName());
    }

    @Before
    public void setup(){
        Nd4j.setDataType(DataType.FLOAT);
    }

    @After
    public void tearDown() {
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(false);
    }

    private static ExecutorConfiguration configuration = ExecutorConfiguration.builder()
            .executionMode(ExecutionMode.SEQUENTIAL)
            .profilingMode(OpExecutioner.ProfilingMode.DISABLED)
            .gatherTimings(true)
            .outputMode(OutputMode.VARIABLE_SPACE)
            .build();

    protected static List<Object[]> fetchTestParams(String baseDir, String modelFileName, ExecuteWith executeWith, File localTestDir) throws IOException {
        String[] modelNames = modelDirNames(baseDir, executeWith, modelFileName);
        List<Object[]> modelParams = new ArrayList<>();
        for (int i = 0; i < modelNames.length; i++) {
            Object[] currentParams = new Object[4];
            currentParams[0] = inputVars(modelNames[i], baseDir, localTestDir); //input variable map - could be null
            currentParams[1] = outputVars(modelNames[i], baseDir, localTestDir); //saved off predictions
            currentParams[2] = modelNames[i];
            currentParams[3] = localTestDir;
            modelParams.add(currentParams);
        }
        return modelParams;
    }

    protected static void checkOnlyOutput(Map<String, INDArray> inputs, Map<String, INDArray> predictions, String modelName,
                                          String baseDir, String modelFilename, ExecuteWith execType, BiFunction<File,String,SameDiff> loader,
                                          Double maxRelErrorOverride, Double minAbsErrorOverride, boolean printArraysDebugging) throws IOException {
        Preconditions.checkArgument((maxRelErrorOverride == null) == (minAbsErrorOverride == null), "Both maxRelErrorOverride and minAbsErrorOverride" +
                " must be null or both must be provided");
        Nd4j.EPS_THRESHOLD = 1e-3;

        Set<String> outputsToCheck = new HashSet<>();
        for(String s : predictions.keySet()) {
            // we need to convert name from python name format with . on indices, to :. i.e.: output.1 -> output:1
            if (s.matches(".*\\.\\d+")) {
                int idx = s.lastIndexOf('.');
                s = s.substring(0, idx) + ":" + s.substring(idx + 1);
            }
            outputsToCheck.add(s);
        }

        Pair<SameDiff,Map<String,INDArray>> p = getGraphAfterExec(baseDir, modelFilename, modelName, inputs, execType, loader, null, outputsToCheck, printArraysDebugging);
        SameDiff graph = p.getFirst();
        Map<String,INDArray> sameDiffPredictions = p.getSecond();
//        SameDiff graph = graphLoaderFunction.apply(new ClassPathResource(baseDir + "/" + modelName + "/" + modelFilename).getFile(), modelName);
        //Collect coverage info about ops
       /* TensorflowFrameworkImporter tensorflowFrameworkImporter = new TensorflowFrameworkImporter();
        File oldModel = new ClassPathResource(baseDir + "/" + modelName + "/" + modelFilename).getFile();
        GraphDef g  = GraphDef.parseFrom(IOUtils.toByteArray(oldModel.toURI()));
        TensorflowIRGraph tensorflowIRGraph = new TensorflowIRGraph(g,tensorflowFrameworkImporter.getOpDefList(),tensorflowFrameworkImporter.getRegistry());
        TensorflowIRGraphRunner tensorflowIRGraphRunner = new TensorflowIRGraphRunner(tensorflowIRGraph,p.getFirst().inputs(),outputsToCheck.stream().collect(Collectors.toList()));
        Map<String,INDArray> outputs = tensorflowIRGraphRunner.run(inputs);
        SameDiff oldForComparison = TFGraphMapper.importGraph(oldModel);
        Map<String,INDArray> oldOutputs = oldForComparison.outputAll(inputs);
        */
        OpValidation.collectTensorflowImportCoverage(graph);

        if (!execType.equals(ExecuteWith.JUST_PRINT)) {
            assertTrue("No predictions to validate", predictions.keySet().size() > 0);
            for (String outputNode : predictions.keySet()) {
                INDArray nd4jPred = null;
                INDArray tfPred = null;

                String nd4jNode = outputNode;

                // we need to convert name from python name format with . on indices, to :. i.e.: output.1 -> output:1
                if (outputNode.contains("."))
                    nd4jNode = outputNode.replaceAll("\\.", ":");

                try {
                    nd4jPred = sameDiffPredictions.get(nd4jNode);
                } catch (NullPointerException e) {
                    throw new NullPointerException("Can't find SameDiff variable with name [" + nd4jNode + "]");
                }

                try {
                    tfPred = predictions.get(outputNode);
                } catch (NullPointerException e) {
                    throw new NullPointerException("Can't find predicted variable with name [" + outputNode + "]");
                }

                assertNotNull(nd4jPred);
                assertNotNull(tfPred);

                if(maxRelErrorOverride == null) {
                    long[] sTf = tfPred.shape();
                    long[] sNd4j = nd4jPred.shape();
                    assertArrayEquals("Shapes for node \"" + outputNode + "\" are not equal: TF: " + Arrays.toString(sTf) + " vs SD: " + Arrays.toString(sNd4j), sTf, sNd4j);

                    // TODO: once we add more dtypes files - this should be removed
                    if (tfPred.dataType() != nd4jPred.dataType())
                        nd4jPred = nd4jPred.castTo(tfPred.dataType());

                    boolean eq = getEqualityFunction(modelName, outputNode, tfPred, nd4jPred).apply(tfPred, nd4jPred);

                    if(!eq){
                        //Check for both NaN, both inf
                        if(tfPred.dataType().isFPType() && tfPred.equalShapes(nd4jPred) && tfPred.isNaN().castTo(DataType.INT).sumNumber().intValue() == tfPred.length()
                                && nd4jPred.isNaN().castTo(DataType.INT).sumNumber().intValue() == nd4jPred.length()){
                            //All NaNs in both arrays
                            eq = true;
                        } else if(tfPred.dataType().isFPType() && tfPred.equalShapes(nd4jPred) && tfPred.isInfinite().castTo(DataType.INT).sumNumber().intValue() == tfPred.length()
                                && nd4jPred.isInfinite().castTo(DataType.INT).sumNumber().intValue() == nd4jPred.length()){
                            //All infinite in both arrays. But need to check that it's all positive vs. negative infinite in both cases...
                            NdIndexIterator iter = new NdIndexIterator(tfPred.shape());
                            eq = true;
                            while(iter.hasNext()) {
                                long[] next = iter.next();
                                //Already know they are both infinite, only question is whether they are both positive and negative
                                double d1 = tfPred.getDouble(next);
                                double d2 = nd4jPred.getDouble(next);
                                if((d1 > 0) != (d2 > 0)){
                                    eq = false;
                                    break;
                                }
                            }
                        }

                        if(!eq) {
                            NDArrayStrings s = new NDArrayStrings();
                            String s1 = s.format(tfPred, false);
                            String s2 = s.format(nd4jPred, false);
                            System.out.print("TF: ");
                            System.out.println(tfPred.toStringFull());
                            System.out.print("SD: ");
                            System.out.println(nd4jPred.toStringFull());
                        }
                    }

                    assertTrue("Predictions do not match on " + modelName + ", node " + outputNode, eq);
                } else {

                    if(!tfPred.equalShapes(nd4jPred)) {
                        fail("Output node \"" + outputNode + "\" SameDiff output shape does not match TF output shape: SameDiff shape: " +
                                Arrays.toString(nd4jPred.shape()) + " vs. TF shape: " + Arrays.toString(tfPred.shape()));
                    }

                    if(tfPred.dataType() != nd4jPred.dataType()) {
                        fail("Output node \"" + outputNode + "\" SameDiff output datatype does not match TF output : SameDiff type: " +
                                nd4jPred.dataType() + " vs. TF datatype: " + tfPred.dataType());
                    }

                    if(!tfPred.dataType().isFPType()){
                        //Can't do relative error on long type...
                        tfPred = tfPred.castTo(DataType.DOUBLE);
                        nd4jPred = nd4jPred.castTo(DataType.DOUBLE);
                    }

                    INDArray diff = Transforms.abs(tfPred.sub(nd4jPred), false);
                    INDArray absErrorMask = diff.gte(minAbsErrorOverride).castTo(tfPred.dataType());   //value 1 if x[i] > minAbsError; value 0 otherwise. Used to get rid of 1e-30 vs. 1e-29 type failures
                    INDArray sumAbs = Transforms.abs(tfPred, true).addi(Transforms.abs(nd4jPred, true));
                    BooleanIndexing.replaceWhere(sumAbs, 1.0, Conditions.equals(0.0));  //Can only get 0.0 if both are zeros - need to avoid 0/0=NaN
                    INDArray relError = diff.divi(sumAbs);
                    relError.muli(absErrorMask);


                    /*
                    Try to detect bad test.
                    The idea: suppose all values are small, and are excluded due to minAbsError threshold
                    i.e., all 1e-5 vs. -1e-5 with min abs error of 1e-4
                    */
                    //TODO FIX ME
                    INDArray maxAbs = Transforms.max(Transforms.abs(tfPred.castTo(DataType.DOUBLE), true), Transforms.abs(nd4jPred.castTo(DataType.DOUBLE), true), true);
                    long countMaxAbsGTThreshold = maxAbs.gte(minAbsErrorOverride).castTo(DataType.INT).sumNumber().intValue();
                    long countNotMasked = absErrorMask.sumNumber().intValue();  //Values are 0 or 1... if all 0s -> nothing being tested
                    if(countNotMasked == 0 && countMaxAbsGTThreshold == 0){
                        fail("All values for node " + outputNode + " are masked out due to minAbsError=" + minAbsErrorOverride +
                                " and max values are all less than minAbsError - nothing can be tested here");
                    }

                    int countExceeds = Nd4j.getExecutioner().exec(new MatchCondition(relError, Conditions.greaterThan(maxRelErrorOverride))).getInt(0);

                    double maxRE = -1;
                    if(countExceeds > 0){
                        maxRE = relError.maxNumber().doubleValue();
                    }


                    assertEquals( outputNode + ": " + countExceeds + " values exceed maxRelError=" + maxRelErrorOverride
                            + " with minAbsError=" + minAbsErrorOverride + "; largest observed relError=" + maxRE, 0, countExceeds);
                }
            }
            log.info("TEST {} PASSED with {} arrays compared...", modelName, predictions.keySet().size());
        }

        //Serialize and deserialize, check equality:
        ByteBuffer serialized = graph.asFlatBuffers(true);
        Preconditions.checkNotNull(serialized, "Serialization failed? Null output");
        OpValidation.checkDeserializedEquality(graph, serialized, new TestCase(graph).testName(modelName).placeholderValues(inputs));


        Nd4j.EPS_THRESHOLD = 1e-5;
    }

    public static void checkIntermediate(Map<String, INDArray> inputs, String modelName, String baseDir, String modelFileName,
                                         ExecuteWith execType, File localTestDir, boolean printArraysDebugging) throws IOException {
        checkIntermediate(inputs, modelName, baseDir, modelFileName, execType, LOADER, null, null, localTestDir, printArraysDebugging);
    }

    public static void checkIntermediate(Map<String, INDArray> inputs, String modelName, String baseDir, String modelFileName,
                                         ExecuteWith execType, BiFunction<File,String,SameDiff> loader,
                                         Double maxRelErrorOverride, Double minAbsErrorOverride, File localTestDir, boolean printArraysDebugging) throws IOException {
        Preconditions.checkArgument((maxRelErrorOverride == null) == (minAbsErrorOverride == null), "Both maxRelErrorOverride and minAbsErrorOverride" +
                " must be null or both must be provided");
        Nd4j.EPS_THRESHOLD = 1e-3;
        OpExecOrderListener listener = new OpExecOrderListener();       //Used to collect exec order
        Pair<SameDiff, Map<String,INDArray>> p = getGraphAfterExec(baseDir, modelFileName, modelName, inputs, execType, loader, Collections.singletonList(listener), null, printArraysDebugging);
        SameDiff graph = p.getFirst();
        Map<String,INDArray> sdPredictions = p.getSecond();

        //Collect coverage info about ops
        OpValidation.collectTensorflowImportCoverage(graph);

        if (!execType.equals(ExecuteWith.JUST_PRINT)) {
            int count = 0;
            //Evaluate the nodes in their execution order - this is useful for debugging (as we want the *first* failure
            // to be detected before later failures)
            List<String> varNames = new ArrayList<>();
            Map<String,SameDiffOp> fns = graph.getOps();
            List<String> execOrder = listener.getOpNamesList();
            for(String opName : execOrder){
                String[] outputs = graph.getOutputsForOp(fns.get(opName).getOp());
                Collections.addAll(varNames, outputs);
            }

            for (String varName : varNames) {
                if (!inputs.containsKey(varName)) { //avoiding placeholders
                    INDArray tfValue = intermediateVars(modelName, baseDir, varName, localTestDir);
                    if (tfValue == null) {
                        continue;
                    }
                    log.info("Starting check: variable {}", varName);
                    if (skipNode(modelName, varName)) {
                        log.info("\n\tFORCING no check on " + varName);
                    } else {
                        //assertArrayEquals("Shape not equal on node " + varName, tfValue.shape(), graph.getVariable(varName).getShape());
                        INDArray sdVal = sdPredictions.get(varName);
                        if(maxRelErrorOverride != null){
                            INDArray diff = Transforms.abs(tfValue.sub(sdVal), false);
                            INDArray absErrorMask = diff.gte(minAbsErrorOverride);   //value 1 if x[i] > minAbsError; value 0 otherwise. Used to get rid of 1e-30 vs. 1e-29 type failures
                            INDArray sumAbs = Transforms.abs(tfValue, true).addi(Transforms.abs(sdVal, true));
                            BooleanIndexing.replaceWhere(sumAbs, 1.0, Conditions.equals(0.0));  //Can only get 0.0 if both are zeros - need to avoid 0/0=NaN
                            INDArray relError = diff.divi(sumAbs);
                            relError.muli(absErrorMask);

                            int countExceeds = Nd4j.getExecutioner().exec(new MatchCondition(relError, Conditions.greaterThan(maxRelErrorOverride))).getInt(0);

                            double maxRE = -1;
                            //Mainly used for analysis in debugger:
                            DifferentialFunction op = null;
                            String[] opInputs = null;
                            if(countExceeds > 0){
                                maxRE = relError.maxNumber().doubleValue();
                                //Find the op that this variable is produced by
                                op = graph.getVariableOutputOp(varName);
                                opInputs = graph.getInputsForOp(op);
                            }


                            assertEquals( varName + ": " + countExceeds + " values exceed maxRelError=" + maxRelErrorOverride
                                    + " with minAbsError=" + minAbsErrorOverride + "; largest observed relError=" + maxRE, 0, countExceeds);
                        } else {
//                            assertEquals("Value not equal on node " + varName, tfValue, sdVal);
                            if(tfValue.equals(sdVal)){
                                System.out.println("Pass: " + varName);
                            } else {
                                System.out.println("FAIL: " + varName);
                                System.out.println("TF:\n" + tfValue);
                                System.out.println("SD:\n" + sdVal);
                            }

                        }
                        log.info("Values and shapes equal for {}", varName);
                        count++;
                    }

                }
            }

            assertTrue("No intermediate variables were checked", count > 0);
        }

        Nd4j.EPS_THRESHOLD = 1e-5;
    }

    public static Pair<SameDiff, Map<String,INDArray>> getGraphAfterExec(String baseDir, String modelFilename, String modelName, Map<String, INDArray> inputs,
                                                                         ExecuteWith executeWith, BiFunction<File,String,SameDiff> graphLoaderFunction, List<Listener> listeners,
                                                                         Set<String> requiredOutputs, boolean printArraysDebugging) throws IOException {
        log.info("RUNNING TEST {}...", modelName);
        SameDiff graph = graphLoaderFunction.apply(new ClassPathResource(baseDir + "/" + modelName + "/" + modelFilename).getFile(), modelName);
        if(listeners != null){
            graph.setListeners(listeners);
        }

        if(printArraysDebugging) {
            graph.addListeners(new ExecPrintListener());
        }

        if(requiredOutputs == null){
            requiredOutputs = graph.variableMap().keySet();
        }

        Map<String,INDArray> outMap = null;
        if (executeWith.equals(ExecuteWith.SAMEDIFF)) {
            //Set memory manager - check that all arrays (other than the ones we requested as output)
            CloseValidationMemoryMgr mmgr = new CloseValidationMemoryMgr(graph, new ArrayCloseMemoryMgr());
           /* long tid = Thread.currentThread().getId();
            if(!graph.getSessions().containsKey(tid))
                graph.getSessions().put(tid, new InferenceSession(graph));*/
            //Execute
           // graph.getSessions().get(tid).setMmgr(mmgr);
            Map<String,String> shapes = new HashMap<>();
            inputs.entrySet().stream().forEach(entry -> {
                shapes.put(entry.getKey(),Arrays.toString(entry.getValue().shape()));
            });

            log.info("Testing inputs with names " + inputs.keySet() + " and shapes " + shapes);

            outMap = graph.output(inputs, new ArrayList<>(requiredOutputs));

            //Check that all arrays were released
            //mmgr.assertAllReleasedExcept(outMap.values());
            graph.getSessions().clear();
        } else if (executeWith.equals(ExecuteWith.LIBND4J)) {
            for (String input : inputs.keySet()) {
                graph.associateArrayWithVariable(inputs.get(input), graph.variableMap().get(input));
            }

//            val string = graph.asFlatPrint();
//            log.info("Graph structure: \n{}", string);
            val executioner = new NativeGraphExecutioner();
            val results = executioner.executeGraph(graph, configuration);

        } else if (executeWith.equals(ExecuteWith.JUST_PRINT)) {
            for (String input : inputs.keySet()) {
                graph.associateArrayWithVariable(inputs.get(input), graph.variableMap().get(input));
            }

            val string = graph.asFlatPrint();
            log.info("Graph structure: \n{}", string);
        }

        return new Pair<>(graph, outMap);
    }

    private static String[] modelDirNames(String base_dir, ExecuteWith executeWith, String modelFileName) throws IOException {
        ResourcePatternResolver resolver = new PathMatchingResourcePatternResolver(new ClassPathResource(base_dir).getClassLoader());
        Resource[] resources = resolver.getResources("classpath*:" + base_dir + "/**/" + modelFileName );
        String[] exampleNames = new String[resources.length];
        for (int i = 0; i < resources.length; i++) {
            String nestedName = resources[i].getURL().toString().split(base_dir + "/")[1];
            exampleNames[i] = nestedName.replaceAll(Pattern.quote(base_dir), "").replaceAll("/" + modelFileName, "");
        }
        return exampleNames;
    }

    protected static Map<String, INDArray> inputVars(String modelName, String base_dir, File localTestDir) throws IOException {
        return readVars(modelName, base_dir, "**.placeholder", true, localTestDir);
    }


    protected static Map<String, INDArray> outputVars(String modelName, String base_dir, File localTestDir) throws IOException {
        return readVars(modelName, base_dir, "**.prediction", true, localTestDir);
    }

    protected static Map<String, INDArray> inbetweenVars(String modelName, String base_dir, File localTestDir) throws IOException {
        return readVars(modelName, base_dir, "**.prediction_inbw", true, localTestDir);
    }


    //return readVars(modelName, base_dir, "**.prediction_inbw", true);

    /**
     * Possible for a single node to give multiple outputs
     *
     * How is a node that has a list of outputs like in the case of "node_multiple_out" work
     * Below is hardcoded for a single node
     */
    protected static INDArray intermediateVars(String modelName, String base_dir, String varName, File localTestDir) throws IOException {
        //convert varName to convention used in naming files
        // "/" replaced by "____"; followed by a digit indicating the output number followed by prediction_inbw.(shape|csv)
        if (varName.contains(":")) {
            varName = varName.replace(':', '.');
        } else {
            varName = varName + ".0";
        }
        String name = varName.replaceAll("/", "____") + ".prediction_inbw";
        Map<String, INDArray> nodeSepOutput = readVars(modelName, base_dir, name, true, localTestDir);

        boolean importNameWorkaround = false;
        if(nodeSepOutput.isEmpty()){
            //Edge case: intermediates were generated with help of import_graph_def method, which by default adds "import/" to names
            // for some reason. https://www.tensorflow.org/api_docs/python/tf/graph_util/import_graph_def
            //So many of earlier intermediate nodes test data were generated with filenames like "import___X..." instead of "X..."
            name = "import____" + name;
            nodeSepOutput = readVars(modelName, base_dir, name, true, localTestDir);
            importNameWorkaround = true;
        }

        //required check for pattern matching as there are scopes and "*" above is a greedy match
        Set<String> removeList = confirmPatternMatch(nodeSepOutput.keySet(), importNameWorkaround ? "import/" + varName : varName);
        for (String toRemove : removeList) {
            nodeSepOutput.remove(toRemove);
        }
        if(importNameWorkaround){
            return nodeSepOutput.get("import/" + varName); //this *should* return a list of the indarrays for each node
        } else {
            return nodeSepOutput.get(varName); //this *should* return a list of the indarrays for each node
        }
    }

    public static Set<String> confirmPatternMatch(Set<String> setOfNames, String varName) {
        Set<String> removeList = new HashSet<>();
        for (String name : setOfNames) {
            if (name.equals(varName)) continue;
            String[] splitByPeriod = name.split("\\.");
            //not a number - maybe another variable deeper in the same scope
            if (!NumberUtils.isNumber(splitByPeriod[splitByPeriod.length - 1])) {
                removeList.add(name);
            } else if (!String.join(".", Arrays.copyOfRange(splitByPeriod, 0, splitByPeriod.length - 1)).equals(varName)) {
                removeList.add(name);
            }
        }
        return removeList;
    }


    protected static Map<String, INDArray> readVars(String modelName, String base_dir, String pattern, boolean recursive, File localTestDir) throws IOException {
        Map<String, INDArray> varMap = new HashMap<>();
        String modelDir = base_dir + "/" + modelName;

        // key is variable name, value is data type
        val dtypes = new HashMap<String, DataType>();

        List<Pair<Resource,Resource>> resources = new ArrayList<>();
        if(recursive){
            String nameRegex = pattern.replace("**.",".*\\.") + "\\.shape";
//            File baseDir = new File(System.getProperty("java.io.tmpdir"), UUID.randomUUID().toString() + "/" + modelName);
//            baseDir.mkdirs();
//            baseDir.deleteOnExit();
//            new ClassPathResource(modelDir).copyDirectory(baseDir);

            // checking out, if local folder declared
            String localPath = System.getenv(TFGraphTestAllHelper.resourceFolderVar);
            if(localPath != null && (!localPath.contains("src/main/resources") && !localPath.contains("src\\main\\resources"))){
                localPath = FilenameUtils.concat(localPath, "src/main/resources");
            }

            // baseDir will differ, depending on run mode
            File baseDir = localPath == null ? new File(localTestDir, "extracted/" + modelName) : new File(localPath, base_dir + "/" + modelName);
            String[] arr = baseDir.list();

            if(!baseDir.exists() || arr == null || arr.length == 0){
                // we're skipping extraction if we're using local copy of dl4j-tests-resources
                if (localPath == null) {
                    baseDir.mkdirs();
                    baseDir.deleteOnExit();
                    String md = modelDir;
                    if(!md.endsWith("/") && !md.endsWith("\\")){
                        md = md + "/";
                    }
                    new ClassPathResource(md).copyDirectory(baseDir);
                } else{
                    throw new IllegalStateException("local directory declared but could not find files: " + baseDir.getAbsolutePath());
                }

            }

            LinkedList<File> queue = new LinkedList<>();
            queue.add(baseDir);

            while(!queue.isEmpty()){
                File subdir = queue.remove();
                File[] files = subdir.listFiles();
                if (files != null) {
                    for (File f : files) {
                        if (f.isDirectory()) {
                            queue.add(f);
                        } else {
                            String filename = f.getName();
                            if(filename.matches(nameRegex)){
                                File csvFile = new File(f.getAbsolutePath().replace(".shape",".csv"));
                                resources.add(new Pair<>(new FileSystemResource(f), new FileSystemResource(csvFile)));
                            } else if (filename.equals("dtypes")) {
                                List<String> stringList;

                                try (val is = new BufferedInputStream(new FileInputStream(f))) {
                                    stringList = IOUtils.readLines(is, StandardCharsets.UTF_8);

                                    for (val s:stringList) {
                                        val split = s.split("\\ ");

                                        val okey = split[0].replaceAll("____", "/");
                                        // adopt / in names
                                        val key = modelDir + "/" + okey;

                                        // parse type directly
                                        DataType value = ArrayOptionsHelper.dataType(split[1]);

                                        // adding key directly
                                        //if (dtypes.containsKey(key))
                                        //    throw new ND4JIllegalStateException("Specified key already exist: [" + key + "]");
                                        //else

                                        dtypes.put(key, value);

                                        // adding zero output duplicate (if it doesn't exist)
                                        if (key.endsWith(".0")) {
                                            val nkey = key.replaceAll("\\.0$","");
                                            if (!dtypes.containsKey(nkey)) {
                                                dtypes.put(nkey, value);
                                            }
                                        } else if (key.endsWith(":0")) {
                                            val nkey = key.replaceAll(":0$","");
                                            if (!dtypes.containsKey(nkey)) {
                                                dtypes.put(nkey, value);
                                            }
                                        }
                                    }
                                } catch (FileNotFoundException e) {
                                    stringList = new ArrayList<>();
                                }
                            }
                        }
                    }
                }
            }
        } else {
            ResourcePatternResolver resolver = new PathMatchingResourcePatternResolver(new ClassPathResource(modelDir).getClassLoader());
            Resource[] r = resolver.getResources("classpath*:" + modelDir + "/" + pattern + ".shape");
            for(Resource res : r){
                String fileName = res.getFilename();
                String varPath = modelDir + "/" + fileName;
                Resource r2 = new org.springframework.core.io.ClassPathResource(varPath.replace(".shape", ".csv"));
                resources.add(new Pair<>(res, r2));
            }

        }

//        Preconditions.checkState(!dtypes.isEmpty(), "No datatypes file was found");

        val dtype = Nd4j.dataType();
        for (int i = 0; i < resources.size(); i++) {
            URI u = resources.get(i).getFirst().getURI();
            String varName = u.toString();
            int idx = varName.indexOf(modelName);
            varName = varName.substring(idx + modelName.length()+1);    //+1 for "/"
            varName = varName.replaceAll("____","/");
            varName = varName.replaceAll(".placeholder.shape","");
            varName = varName.replaceAll(".prediction.shape","");
            varName = varName.replaceAll(".prediction_inbw.shape","");

            DataType type = dtypes.get(modelDir + "/" + varName);

            List<String> lines; //= FileUtils.readLines(new ClassPathResource(varPath).getFile(), Charset.forName("UTF-8"));
            try(InputStream is = new BufferedInputStream(resources.get(i).getFirst().getInputStream())){
                lines = IOUtils.readLines(is, StandardCharsets.UTF_8);
            }
            List<String> filtered = new ArrayList<>(lines.size());
            for(String s : lines){
                String trimmed = s.trim();
                if(!trimmed.isEmpty()){
                    filtered.add(trimmed);
                }
            }

            if(type == null){
                log.warn("DATATYPE NOT AVAILABLE FOR: {} - {}", modelName, varName);
                //Soon: this will be an exception
                type = DataType.FLOAT;
            }

            INDArray varValue;
            if(filtered.size() == 0){
                //Scalar
                String content = IOUtils.toString(resources.get(i).getSecond().getInputStream(), StandardCharsets.UTF_8);
                switch (type){
                    case DOUBLE:
                    case FLOAT:
                    case HALF:
                    case BFLOAT16:
                        varValue = Nd4j.scalar(type, parseDouble(content));
                        break;
                    case LONG:
                    case INT:
                    case SHORT:
                    case UBYTE:
                    case BYTE:
                    case UINT16:
                    case UINT32:
                    case UINT64:
                        varValue = Nd4j.scalar(type, parseLong(content));
                        break;
                    case BOOL:
                        varValue = Nd4j.scalar(parseBoolean(content));
                        break;
                    case UTF8:
                        varValue = Nd4j.scalar(content);
                        break;
                    case COMPRESSED:
                    case UNKNOWN:
                    default:
                        throw new UnsupportedOperationException("Unknown / not implemented datatype: " + type);
                }
            } else {
                int[] varShape = new int[filtered.size()];
                for( int j=0; j<filtered.size(); j++ ){
                    varShape[j] = Integer.parseInt(filtered.get(j));
                }

                try {
                    String content;
                    Pair<Resource,Resource> p = resources.get(i);
                    boolean isRef = p.getSecond().isFile() && !p.getSecond().exists();

                    InputStream stream;
                    if(isRef){
                        //Slight hack for loading strumpf reference files
                        File r = new StrumpfResolver().localCacheRoot();
                        String path = p.getSecond().getFile() + StrumpfResolver.REF;
                        File f = ResourceFile.fromFile(path).localFile(r);
                        stream = new BufferedInputStream(new FileInputStream(f));
                    } else {
                        stream = new BufferedInputStream(resources.get(i).getSecond().getInputStream());
                    }

                    try(InputStream is = stream){
                        content = String.join("\n", IOUtils.readLines(is, StandardCharsets.UTF_8));
                    }

                    if (content.isEmpty()) {
                        //Should be zeros in shape
                        boolean foundZero = false;
                        for( int s : varShape){
                            foundZero |= (s == 0);
                        }
                        if(foundZero){
                            varValue = Nd4j.create(type, ArrayUtil.toLongArray(varShape));
                        } else {
                            throw new IllegalStateException("Empty data but non-empty shape: " + resources.get(i).getSecond());
                        }
                    } else {
                        if(varShape.length == 1 && varShape[0] == 0)        //Annoyingly, some scalars have shape [0] instead of []
                            varShape = new int[0];

                        String[] cLines = content.split("\n");
                        switch (type){
                            case DOUBLE:
                            case FLOAT:
                            case HALF:
                            case BFLOAT16:
                                double[] dArr = new double[cLines.length];
                                int x=0;
                                while(x < dArr.length){
                                    dArr[x] = parseDouble(cLines[x]);
                                    x++;
                                }
                                varValue = Nd4j.createFromArray(dArr).castTo(type).reshape('c', varShape);
                                break;
                            case LONG:
                            case INT:
                            case SHORT:
                            case UBYTE:
                            case BYTE:
                            case UINT16:
                            case UINT32:
                            case UINT64:
                                long[] lArr = new long[cLines.length];
                                int y=0;
                                while(y < lArr.length){
                                    lArr[y] = parseLong(cLines[y]);
                                    y++;
                                }
                                varValue = Nd4j.createFromArray(lArr).castTo(type).reshape('c', varShape);
                                break;
                            case BOOL:
                                boolean[] bArr = new boolean[cLines.length];
                                int z=0;
                                while(z < bArr.length){
                                    bArr[z] = parseBoolean(cLines[z]);
                                    z++;
                                }
                                varValue = Nd4j.createFromArray(bArr).reshape('c', varShape);
                                break;
                            case UTF8:
                                varValue = Nd4j.create(cLines).reshape('c', varShape);
                                break;
                            case COMPRESSED:
                            case UNKNOWN:
                            default:
                                throw new UnsupportedOperationException("Unknown / not implemented datatype: " + type);
                        }
                    }
                } catch (NumberFormatException e) {
                    log.warn("Error parsing number", e);
                    continue;
                }
            }

            varMap.put(varName, varValue);
        }
        return varMap;
    }

    private static long parseLong(String line){
        line = line.trim();       //Handle whitespace
        if(line.matches("-?\\d+\\.0+")){
            //Annoyingly, some integer data is stored with redundant/unnecessary zeros - like "-7.0000000"
            return Long.parseLong(line.substring(0, line.indexOf('.')));
        } else {
            return Long.parseLong(line);
        }
    }

    private static double parseDouble(String line){
        line = line.trim();   //Handle whitespace - some lines are like "      -inf"
        if("nan".equalsIgnoreCase(line)){
            return Double.NaN;
        } else if("inf".equalsIgnoreCase(line)) {
            return Double.POSITIVE_INFINITY;
        } else if("-inf".equalsIgnoreCase(line)){
            return Double.NEGATIVE_INFINITY;
        } else {
            return Double.parseDouble(line);
        }
    }

    private static boolean parseBoolean(String line){
        line = line.trim();
        if(line.matches("1(\\.0*)?")){          //Booleans are ocassionally represented like 1.000000 or 0.000000
            return true;
        } else if(line.matches("0(\\.0*)?")){
            return false;
        }
        return Boolean.parseBoolean(line);
    }


    public static Pair<Double,Double> testPrecisionOverride(String testName){
        if("conv_4".equalsIgnoreCase(testName)) {
            //Most values: around 1k. So this is the 6th significant figure, which is OK
            return new Pair<>(1e-3, 1e-5);
        }
        return null;
    }

    public static boolean equalsWithEps(double a, double b){
        return Math.abs(a - b) <= 0.00001;
    }

    public static BiFunction<INDArray, INDArray, Boolean> getEqualityFunction(String modelName, String varName, INDArray tf, INDArray sd){
        if(modelName.startsWith("topk")){
            return (t, s) -> Nd4j.sort(t, true).equals(Nd4j.sort(s, true));
        }

        if(modelName.startsWith("empty")){
            return (t, s) -> {
                boolean areEqualShapes = t.equalShapes(s);
                boolean areEqualDataTypes = t.dataType() == s.dataType();
                return areEqualShapes && areEqualDataTypes;
            };        }

        // sum of all elements along dimesions before and after shuffle has to be the same
        if(modelName.startsWith("random_shuffle")){
            return (t, s) -> Nd4j.sort(t, true).equals(Nd4j.sort(s, true));
        }

        if(modelName.startsWith("random_normal")){
            return (t, s) -> {
                boolean areEqualShapes = t.equalShapes(s);
                double meanS = s.meanNumber().doubleValue();
                double meanT = t.meanNumber().doubleValue();
                double stdS = s.stdNumber().doubleValue();
                double stdT = t.stdNumber().doubleValue();
                double eps = 1;
                return areEqualShapes && (Math.abs(meanS-meanT) < eps) && (Math.abs(stdS-stdT) < eps);
            };        }

        if(modelName.startsWith("random_gamma")){
            return (t, s) -> {
                boolean areEqualShapes = t.equalShapes(s);
                boolean nonNegativeValues = (t.minNumber().doubleValue() > 0) && (t.minNumber().doubleValue() > 0);
                double meanS = s.meanNumber().doubleValue();
                double meanT = t.meanNumber().doubleValue();
                double stdS = s.stdNumber().doubleValue();
                double stdT = t.stdNumber().doubleValue();
                double eps = 1;
                return areEqualShapes && nonNegativeValues && (Math.abs(meanS-meanT) < eps) && (Math.abs(stdS-stdT) < eps);
            };
        }

        if(modelName.startsWith("random_poisson") || modelName.startsWith("random_poisson_v2")){
            return (t, s) -> {
                boolean areEqualShapes = t.equalShapes(s);
                boolean nonNegativeValues = (t.minNumber().doubleValue() >= 0) && (t.minNumber().doubleValue() >= 0);
                double meanS = s.meanNumber().doubleValue();
                double meanT = t.meanNumber().doubleValue();
                double stdS = s.stdNumber().doubleValue();
                double stdT = t.stdNumber().doubleValue();
                double eps = 1;
                return areEqualShapes && nonNegativeValues && (Math.abs(meanS-meanT) < eps) && (Math.abs(stdS-stdT) < eps);
            };
        }

        if(modelName.startsWith("random_uniform")|| modelName.startsWith("random_uniform_int")){
            return (t, s) -> {
                boolean areEqualShapes = t.equalShapes(s);
                double meanS = s.meanNumber().doubleValue();
                double meanT = t.meanNumber().doubleValue();
                double stdS = s.stdNumber().doubleValue();
                double stdT = t.stdNumber().doubleValue();
                double eps = 1;
                return areEqualShapes && (Math.abs(stdS-stdT) < eps) && (Math.abs(meanS-meanT) < eps);
            };
        }

        if(modelName.startsWith("alpha_dropout") || modelName.startsWith("layers_dropout") || modelName.startsWith("dropout"))
            //We can't compare dropout using simple equality due to randomness
            return (t, s) -> {
                double[] tfNums = t.ravel().toDoubleVector();
                double[] sdNums = s.ravel().toDoubleVector();

                Double seen1 = null, seen2 = null;
                for(int i = 0 ; i < tfNums.length ; i++){
                    if(!equalsWithEps(tfNums[i], sdNums[i])){

                        // if we have only seen one inequality so far, figure out which is the dropout
                        if(seen1 != null && seen2 != null){
                            if(equalsWithEps(tfNums[i], seen1) || equalsWithEps(sdNums[i], seen1)) // the dropout is in seen1
                                seen2 = null;
                            else if(equalsWithEps(tfNums[i], seen2) || equalsWithEps(sdNums[i], seen2)){ // the dropout is in seen2
                                seen1 = seen2;
                                seen2 = null;
                            } else // neither match
                                return false;
                        }

                        if(seen1 != null){
                            if(!equalsWithEps(tfNums[i], seen1) && !equalsWithEps(sdNums[i], seen1))
                                return false;
                        } else {
                            seen1 = tfNums[i];
                            seen2 = sdNums[i];
                        }
                    }
                }

                return true;
            };

        return Object::equals;
    }

}
>>>>>>> KonduitAI-master
