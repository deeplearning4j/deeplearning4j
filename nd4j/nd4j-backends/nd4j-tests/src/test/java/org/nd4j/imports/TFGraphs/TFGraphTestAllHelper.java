/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.imports.TFGraphs;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.builder.Diff;
import org.apache.commons.lang3.math.NumberUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;
import org.nd4j.autodiff.execution.NativeGraphExecutioner;
import org.nd4j.autodiff.execution.conf.ExecutionMode;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.BiFunction;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.string.NDArrayStrings;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.core.io.support.ResourcePatternResolver;

import java.io.*;
import java.net.URI;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.regex.Pattern;

import static org.junit.Assert.*;
import static org.nd4j.imports.TFGraphs.TFGraphsSkipNodes.skipNode;

/**
 * Created by susaneraly on 11/6/17.
 */
@Slf4j
public class TFGraphTestAllHelper {

    public enum ExecuteWith {
        SAMEDIFF, LIBND4J, JUST_PRINT
    }

    public static class DefaultGraphLoader implements BiFunction<File,String,SameDiff> {
        @Override
        public SameDiff apply(File file, String name) {
            try(InputStream is = new BufferedInputStream(new FileInputStream(file))){
                SameDiff sd = TFGraphMapper.getInstance().importGraph(is);
                return sd;
            } catch (IOException e){
                throw new RuntimeException(e);
            }
        }
    }

    public static final DefaultGraphLoader LOADER = new DefaultGraphLoader();

    @BeforeClass
    public void beforeClass(){
        log.info("Starting tests for class: " + getClass().getName());
    }

    @Before
    public void setup(){
        Nd4j.setDataType(DataBuffer.Type.FLOAT);
    }

    @After
    public void tearDown() throws Exception {
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
                                          Double maxRelErrorOverride, Double minAbsErrorOverride) throws IOException {
        Preconditions.checkArgument((maxRelErrorOverride == null) == (minAbsErrorOverride == null), "Both maxRelErrorOverride and minAbsErrorOverride" +
                " must be null or both must be provided");
        Nd4j.EPS_THRESHOLD = 1e-3;

        SameDiff graph = getGraphAfterExec(baseDir, modelFilename, modelName, inputs, execType, loader);

        //Collect coverage info about ops
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
                    nd4jPred = graph.getVariable(nd4jNode).getArr();
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
                    assertArrayEquals("Shapes are not equal: " + Arrays.toString(sTf) + " vs " + Arrays.toString(sNd4j), sTf, sNd4j);
                    boolean eq = tfPred.equals(nd4jPred);
                    if(!eq){
                        NDArrayStrings s = new NDArrayStrings();
                        String s1 = s.format(tfPred, false);
                        String s2 = s.format(nd4jPred, false);
                        System.out.print("TF: ");
                        System.out.println(s1);
                        System.out.print("SD: ");
                        System.out.println(s2);

                    }
                    assertTrue("Predictions do not match on " + modelName + ", node " + outputNode, eq);
                } else {
                    INDArray diff = Transforms.abs(tfPred.sub(nd4jPred), false);
                    INDArray absErrorMask = diff.gte(minAbsErrorOverride);   //value 1 if x[i] > minAbsError; value 0 otherwise. Used to get rid of 1e-30 vs. 1e-29 type failures
                    INDArray sumAbs = Transforms.abs(tfPred, true).addi(Transforms.abs(nd4jPred, true));
                    BooleanIndexing.replaceWhere(sumAbs, 1.0, Conditions.equals(0.0));  //Can only get 0.0 if both are zeros - need to avoid 0/0=NaN
                    INDArray relError = diff.divi(sumAbs);
                    relError.muli(absErrorMask);


                    /*
                    Try to detect bad test.
                    The idea: suppose all values are small, and are excluded due to minAbsError threshold
                    i.e., all 1e-5 vs. -1e-5 with min abs error of 1e-4
                    */
                    INDArray maxAbs = Transforms.max(Transforms.abs(tfPred, true), Transforms.abs(tfPred, true), true);
                    long countMaxAbsGTThreshold = maxAbs.gte(minAbsErrorOverride).sumNumber().intValue();
                    long countNotMasked = absErrorMask.sumNumber().intValue();  //Values are 0 or 1... if all 0s -> nothing being tested
                    if(countNotMasked == 0 && countMaxAbsGTThreshold == 0){
                        fail("All values for node " + outputNode + " are masked out due to minAbsError=" + minAbsErrorOverride +
                                " and max values are all less than minAbsError - nothing can be tested here");
                    }

                    int countExceeds = Nd4j.getExecutioner().exec(new MatchCondition(relError, Conditions.greaterThan(maxRelErrorOverride))).z().getInt(0);

                    double maxRE = -1;
                    if(countExceeds > 0){
                        maxRE = relError.maxNumber().doubleValue();
                    }


                    assertEquals( outputNode + ": " + countExceeds + " values exceed maxRelError=" + maxRelErrorOverride
                            + " with minAbsError=" + minAbsErrorOverride + "; largest observed relError=" + maxRE, 0, countExceeds);
                }
            }
            log.info("\n\tTEST {} PASSED with {} arrays compared...", modelName, predictions.keySet().size());
            log.info("\n========================================================\n");
        }

        Nd4j.EPS_THRESHOLD = 1e-5;
    }

    public static void checkIntermediate(Map<String, INDArray> inputs, String modelName, String baseDir, String modelFileName,
                                         ExecuteWith execType, File localTestDir) throws IOException {
        checkIntermediate(inputs, modelName, baseDir, modelFileName, execType, LOADER, null, null, localTestDir);
    }

    public static void checkIntermediate(Map<String, INDArray> inputs, String modelName, String baseDir, String modelFileName,
                                         ExecuteWith execType, BiFunction<File,String,SameDiff> loader,
                                         Double maxRelErrorOverride, Double minAbsErrorOverride, File localTestDir) throws IOException {
        Preconditions.checkArgument((maxRelErrorOverride == null) == (minAbsErrorOverride == null), "Both maxRelErrorOverride and minAbsErrorOverride" +
                " must be null or both must be provided");
        Nd4j.EPS_THRESHOLD = 1e-3;
        SameDiff graph = getGraphAfterExec(baseDir, modelFileName, modelName, inputs, execType, loader);

        //Collect coverage info about ops
        OpValidation.collectTensorflowImportCoverage(graph);

        if (!execType.equals(ExecuteWith.JUST_PRINT)) {
            int count = 0;
            //Evaluate the nodes in their execution order - this is useful for debugging (as we want the *first* failure
            // to be detected before later failures)
            Set<String> varNamesSet = new HashSet<>(graph.variableMap().keySet());
            List<String> varNames = new ArrayList<>();
            Map<String,DifferentialFunction> fns = graph.getFunctionInstancesById();  //LinkedHashMap defines execution order
            for(Map.Entry<String,DifferentialFunction> e : fns.entrySet()){
                String[] outputs = graph.getOutputsForFunction(e.getValue());
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
                        assertArrayEquals("Shape not equal on node " + varName, tfValue.shape(), graph.getVariable(varName).getShape());
                        INDArray sdVal = graph.getVariable(varName).getArr();
                        if(maxRelErrorOverride != null){
                            INDArray diff = Transforms.abs(tfValue.sub(sdVal), false);
                            INDArray absErrorMask = diff.gte(minAbsErrorOverride);   //value 1 if x[i] > minAbsError; value 0 otherwise. Used to get rid of 1e-30 vs. 1e-29 type failures
                            INDArray sumAbs = Transforms.abs(tfValue, true).addi(Transforms.abs(sdVal, true));
                            BooleanIndexing.replaceWhere(sumAbs, 1.0, Conditions.equals(0.0));  //Can only get 0.0 if both are zeros - need to avoid 0/0=NaN
                            INDArray relError = diff.divi(sumAbs);
                            relError.muli(absErrorMask);

                            int countExceeds = Nd4j.getExecutioner().exec(new MatchCondition(relError, Conditions.greaterThan(maxRelErrorOverride))).z().getInt(0);

                            double maxRE = -1;
                            //Mainly used for analysis in debugger:
                            DifferentialFunction op = null;
                            String[] opInputs = null;
                            if(countExceeds > 0){
                                maxRE = relError.maxNumber().doubleValue();
                                //Find the op that this variable is produced by
                                op = graph.getVariableOutputFunction(varName);
                                opInputs = graph.getInputsForFunction(op);
                            }


                            assertEquals( varName + ": " + countExceeds + " values exceed maxRelError=" + maxRelErrorOverride
                                    + " with minAbsError=" + minAbsErrorOverride + "; largest observed relError=" + maxRE, 0, countExceeds);
                        } else {
                            assertEquals("Value not equal on node " + varName, tfValue, sdVal);
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

    public static SameDiff getGraphAfterExec(String baseDir, String modelFilename, String modelName, Map<String, INDArray> inputs,
                                             ExecuteWith executeWith, BiFunction<File,String,SameDiff> graphLoaderFunction) throws IOException {
        log.info("\n\tRUNNING TEST " + modelName + "...");
        SameDiff graph = graphLoaderFunction.apply(new ClassPathResource(baseDir + "/" + modelName + "/" + modelFilename).getFile(), modelName);
//        = TFGraphMapper.getInstance().importGraph(new ClassPathResource(baseDir + "/" + modelName + "/" + modelFilename).getInputStream());
        //System.out.println(graph.summary());
        if (executeWith.equals(ExecuteWith.SAMEDIFF)) {
            if (!inputs.isEmpty()) {
                graph.execWithPlaceHolder(inputs); //This is expected to be just one result
            } else {
                graph.execAndEndResults(); //there are graphs with no placeholders like g_00
            }
        } else if (executeWith.equals(ExecuteWith.LIBND4J)) {
            for (String input : inputs.keySet()) {
                graph.associateArrayWithVariable(inputs.get(input), graph.variableMap().get(input));
            }

//            val string = graph.asFlatPrint();
//            log.info("Graph structure: \n{}", string);
            val executioner = new NativeGraphExecutioner();
            val results = executioner.executeGraph(graph, configuration);

//            graph.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/conv_0.fb"));
        } else if (executeWith.equals(ExecuteWith.JUST_PRINT)) {
            for (String input : inputs.keySet()) {
                graph.associateArrayWithVariable(inputs.get(input), graph.variableMap().get(input));
            }

            val string = graph.asFlatPrint();
            log.info("Graph structure: \n{}", string);
        }
        return graph;
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
        Map<String, INDArray> nodeSepOutput = readVars(modelName, base_dir, varName.replaceAll("/", "____") + ".prediction_inbw", true, localTestDir);
        //required check for pattern matching as there are scopes and "*" above is a greedy match
        Set<String> removeList = confirmPatternMatch(nodeSepOutput.keySet(), varName);
        for (String toRemove : removeList) {
            nodeSepOutput.remove(toRemove);
        }
        return nodeSepOutput.get(varName); //this *should* return a list of the indarrays for each node
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

        List<Pair<Resource,Resource>> resources = new ArrayList<>();
        if(recursive){
            String nameRegex = pattern.replace("**.",".*\\.") + "\\.shape";
//            File baseDir = new File(System.getProperty("java.io.tmpdir"), UUID.randomUUID().toString() + "/" + modelName);
//            baseDir.mkdirs();
//            baseDir.deleteOnExit();
//            new ClassPathResource(modelDir).copyDirectory(baseDir);
            File baseDir = new File(localTestDir, "extracted/" + modelName);
            String[] arr = baseDir.list();
            if(!baseDir.exists() || arr == null || arr.length == 0){
                baseDir.mkdirs();
                baseDir.deleteOnExit();
                new ClassPathResource(modelDir).copyDirectory(baseDir);
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


        val dtype = Nd4j.dataType();
        for (int i = 0; i < resources.size(); i++) {

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

            INDArray varValue;
            if(filtered.size() == 0){
                //Scalar
                float[] varContents;
                try(InputStream is = new BufferedInputStream(resources.get(i).getSecond().getInputStream())){
                    varContents = Nd4j.readNumpy(is, ",").data().asFloat();
                }
                Preconditions.checkState(varContents.length == 1, "Expected length 1 content for scalar shape; got length %s", varContents.length);
                varValue = Nd4j.trueScalar(varContents[0]);
            } else {
                int[] varShape = new int[filtered.size()];
                for( int j=0; j<filtered.size(); j++ ){
                    varShape[j] = Integer.parseInt(filtered.get(j));
                }

                try {
                    String content;
                    try(InputStream is = new BufferedInputStream(resources.get(i).getSecond().getInputStream())){
//                        long start = System.currentTimeMillis();
//                        log.info("STARTING READ: {}", resources.get(i).getSecond());
                        content = String.join("\n", IOUtils.readLines(is, StandardCharsets.UTF_8));
//                        long end = System.currentTimeMillis();
//                        log.info("Finished reading: {} ms", end - start);
                    }
                    //= FileUtils.readFileToString(new ClassPathResource(varPath.replace(".shape", ".csv")).getFile(), Charset.forName("UTF-8"));
                    if (content.isEmpty()) {
                        if (varShape.length == 1 && varShape[0] == 0) {
                            varValue = Nd4j.empty();
                        } else {
                            throw new IllegalStateException("Empty data but non-empty shape: " + resources.get(i).getSecond());
                        }
                    } else {
                        content = content.replaceAll("False", "0");
                        content = content.replaceAll("True", "1");
                        float[] varContents = Nd4j.readNumpy(new ByteArrayInputStream(content.getBytes(StandardCharsets.UTF_8)), ",").data().asFloat();
                        if (varShape.length == 1) {
                            if (varShape[0] == 0) {
                                varValue = Nd4j.trueScalar(varContents[0]);
                            } else {
                                varValue = Nd4j.trueVector(varContents);
                            }
                        } else {
                            varValue = Nd4j.create(varContents, varShape);
                        }
                    }
                } catch (NumberFormatException e) {
                    log.warn("Error parsing number", e);
                    continue;
                }
            }

            URI u = resources.get(i).getFirst().getURI();
            String varName = u.toString();
            int idx = varName.indexOf(modelName);
            varName = varName.substring(idx + modelName.length()+1);    //+1 for "/"
            varName = varName.replaceAll("____","/");
            varName = varName.replaceAll(".placeholder.shape","");
            varName = varName.replaceAll(".prediction.shape","");
            varName = varName.replaceAll(".prediction_inbw.shape","");
            varMap.put(varName, varValue);
        }
        return varMap;
    }


    public static Pair<Double,Double> testPrecisionOverride(String testName){
        if("conv_4".equalsIgnoreCase(testName)){
            //Most values: around 1k. So this is the 6th significant figure, which is OK
            return new Pair<>(1e-3, 1e-5);
        }
        return null;
    }
}
