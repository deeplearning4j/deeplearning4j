/* ******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.nd4j.autodiff.execution.NativeGraphExecutioner;
import org.nd4j.autodiff.execution.conf.ExecutionMode;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCloseMemoryMgr;
import org.nd4j.autodiff.samediff.internal.memory.CloseValidationMemoryMgr;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.function.BiFunction;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.resources.Resources;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.imports.listeners.ExecPrintListener;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.string.NDArrayStrings;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.core.io.support.ResourcePatternResolver;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.*;
import java.util.regex.Pattern;

import static org.junit.Assert.*;

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
            try(InputStream is = new BufferedInputStream(new FileInputStream(file))){
                SameDiff sd = TFGraphMapper.importGraph(is);
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
                s = s.substring(0, idx) + ":" + s.substring(idx+1);
            }
            outputsToCheck.add(s);
        }

        Pair<SameDiff,Map<String,INDArray>> p = getGraphAfterExec(baseDir, modelFilename, modelName, inputs, execType, loader, null, outputsToCheck, printArraysDebugging);
        SameDiff graph = p.getFirst();
        Map<String,INDArray> sameDiffPredictions = p.getSecond();

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
                    nd4jPred = sameDiffPredictions.get(nd4jNode);
                } catch (NullPointerException e) {
                    throw new NullPointerException("Can't find SameDiff variable with name [" + nd4jNode + "]");
                }

                try {
                    tfPred = predictions.get(outputNode);
                } catch (NullPointerException e) {
                    throw new NullPointerException("Can't find TF predicted variable with name [" + outputNode + "]");
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
                            while(iter.hasNext()){
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
                            System.out.print("TF: ");
                            System.out.println(tfPred.shapeInfoToString());
                            System.out.println(tfPred.toStringFull());
                            System.out.print("SD: ");
                            System.out.println(nd4jPred.shapeInfoToString());
                            System.out.println(nd4jPred.toStringFull());
                        }
                    }
                    assertTrue("Predictions do not match on " + modelName + ", node " + outputNode, eq);
                } else {
                    if(!tfPred.equalShapes(nd4jPred)){
                        fail("Output node \"" + outputNode + "\" SameDiff output shape does not match TF output shape: SameDiff shape: " +
                                Arrays.toString(nd4jPred.shape()) + " vs. TF shape: " + Arrays.toString(tfPred.shape()));
                    }

                    if(tfPred.dataType() != nd4jPred.dataType()){
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

    public static Pair<SameDiff, Map<String,INDArray>> getGraphAfterExec(String baseDir, String modelFilename, String modelName, Map<String, INDArray> inputs,
                                             ExecuteWith executeWith, BiFunction<File,String,SameDiff> graphLoaderFunction, List<Listener> listeners,
                                                                         Set<String> requiredOutputs, boolean printArraysDebugging) throws IOException {
        log.info("RUNNING TEST {}...", modelName);
        File f = Resources.asFile(baseDir + "/" + modelName + "/" + modelFilename);
        SameDiff graph = graphLoaderFunction.apply(f, modelName);
        if(listeners != null){
            graph.setListeners(listeners);
        }

        if(printArraysDebugging){
            graph.addListeners(new ExecPrintListener());
        }

        if(requiredOutputs == null){
            requiredOutputs = graph.variableMap().keySet();
        }

        Map<String,INDArray> outMap = null;
        if (executeWith.equals(ExecuteWith.SAMEDIFF)) {
            //Set memory manager - check that all arrays (other than the ones we requested as output)
            CloseValidationMemoryMgr mmgr = new CloseValidationMemoryMgr(graph, new ArrayCloseMemoryMgr());
            long tid = Thread.currentThread().getId();
            if(!graph.getSessions().containsKey(tid))
                graph.getSessions().put(tid, new InferenceSession(graph));
            //Execute
            graph.getSessions().get(tid).setMmgr(mmgr);
            outMap = graph.output(inputs, new ArrayList<>(requiredOutputs));

            //Check that all arrays were released
            mmgr.assertAllReleasedExcept(outMap.values());
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
        if("conv_4".equalsIgnoreCase(testName)){
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

        if(modelName.startsWith("alpha_dropout") || modelName.startsWith("layers_dropout") || modelName.equals("dropout"))
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
