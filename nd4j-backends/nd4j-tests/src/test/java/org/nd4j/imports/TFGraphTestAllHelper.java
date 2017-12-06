package org.nd4j.imports;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.autodiff.execution.NativeGraphExecutioner;
import org.nd4j.autodiff.execution.conf.ExecutionMode;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.core.io.support.ResourcePatternResolver;

import java.io.IOException;
import java.util.*;

import static org.junit.Assert.assertEquals;

/**
 * Created by susaneraly on 11/6/17.
 */
@Slf4j
@RunWith(Parameterized.class)
public class TFGraphTestAllHelper {

    public enum ExecuteWith {
        SAMEDIFF,
        LIBND4J,
        JUST_PRINT
    }

    //TODO: Later, we can add this as a param so we can test different graphs in samediff and not samediff
    public static final String COMMON_BASE_DIR = "tf_graphs/examples";
    public static final String SAMEDIFF_DEFAULT_BASE_DIR = COMMON_BASE_DIR;
    public static final String LIBND4J_DEFAULT_BASE_DIR = COMMON_BASE_DIR;

    private static ExecutorConfiguration configuration = ExecutorConfiguration.builder()
            .executionMode(ExecutionMode.SEQUENTIAL)
            .profilingMode(OpExecutioner.ProfilingMode.DISABLED)
            .gatherTimings(true)
            .outputMode(OutputMode.IMPLICIT)
            .build();

    protected static List<Object[]> fetchTestParams(ExecuteWith executeWith) throws IOException {
        if (executeWith.equals(ExecuteWith.SAMEDIFF)) {
            return fetchTestParams(SAMEDIFF_DEFAULT_BASE_DIR);
        }
        return fetchTestParams(LIBND4J_DEFAULT_BASE_DIR);
    }

    protected static List<Object[]> fetchTestParams(String baseDir) throws IOException {
        String[] modelNames = modelDirNames(baseDir);
        List<Object[]> modelParams = new ArrayList<>();
        for (int i = 0; i < modelNames.length; i++) {
            Object[] currentParams = new Object[4];
            currentParams[0] = inputVars(modelNames[i], baseDir); //input variable map - could be null
            currentParams[1] = outputVars(modelNames[i], baseDir); //saved off predictions
            currentParams[2] = modelNames[i];
            currentParams[3] = intermediateVars(modelNames[i], baseDir); //intermediate map
            modelParams.add(currentParams);
        }
        return modelParams;
    }

    protected static List<Object[]> fetchTestParams(String baseDir, String modelName) throws IOException {
        List<Object[]> modelParams = new ArrayList<>();
        Object[] currentParams = new Object[4];
        currentParams[0] = inputVars(modelName, baseDir); //input variable map - could be null
        currentParams[1] = outputVars(modelName, baseDir); //saved off predictions
        currentParams[2] = modelName;
        currentParams[3] = intermediateVars(modelName, baseDir); //intermediate map
        modelParams.add(currentParams);
        return modelParams;
    }

    protected static void checkOnlyOutput(Map<String, INDArray> inputs, Map<String, INDArray> predictions, String modelName, ExecuteWith execType) throws IOException {
        if (execType.equals(ExecuteWith.SAMEDIFF)) {
            checkOnlyOutput(inputs, predictions, modelName, SAMEDIFF_DEFAULT_BASE_DIR, execType);
        } else if (execType.equals(ExecuteWith.LIBND4J)) {
            checkOnlyOutput(inputs, predictions, modelName, LIBND4J_DEFAULT_BASE_DIR, execType);
        }
    }

    protected static void checkOnlyOutput(Map<String, INDArray> inputs, Map<String, INDArray> predictions, String modelName, String baseDir, ExecuteWith execType) throws IOException {
        Nd4j.EPS_THRESHOLD = 1e-4;
        INDArray nd4jPred = null;
        Nd4j.getExecutioner().enableDebugMode(true);
        Nd4j.getExecutioner().enableVerboseMode(true);

        val graph = getGraph(baseDir, modelName);

        if (execType.equals(ExecuteWith.SAMEDIFF)) {
            if (!inputs.isEmpty()) {
                graph.execWithPlaceHolder(inputs); //This is expected to be just one result
            } else {
                graph.execAndEndResult("output"); //there are graphs with no placeholders like g_00
            }
            nd4jPred = graph.getVariable("output").getArr();
        } else if (execType.equals(ExecuteWith.LIBND4J)) {
            for (String input : inputs.keySet()) {
                graph.associateArrayWithVariable(inputs.get(input), graph.variableMap().get(input));
            }
            val executioner = new NativeGraphExecutioner();
            val results = executioner.executeGraph(graph, configuration);
            assertEquals(1, results.length); //FIXME: Later
            nd4jPred = graph.getVariable("output").getArr();
            //graph.asFlatFile(new File("../../../libnd4j/tests_cpu/resources/transpose.fb"));
            //return;
        } else if (execType.equals(ExecuteWith.JUST_PRINT)) {
            val string = graph.asFlatPrint();

            log.info("Graph structure: \n{}", string);
            return;
        }

        INDArray tfPred = predictions.get("output");
        assertEquals("Predictions do not match on " + modelName, tfPred.reshape(nd4jPred.shape()), nd4jPred);
        log.info("\n\tTEST " + modelName + " PASSED...");
        log.info("\n========================================================\n");

    }

    public static void checkIntermediate(Map<String, INDArray> inputs, Map<String, INDArray> predictions, Map<String, INDArray[]> intermediates, String modelName, ExecuteWith execType) throws IOException {
        if (execType.equals(ExecuteWith.SAMEDIFF)) {
            checkIntermediate(inputs, predictions, intermediates, modelName, SAMEDIFF_DEFAULT_BASE_DIR, execType);
        } else if (execType.equals(ExecuteWith.LIBND4J)) {
            checkIntermediate(inputs, predictions, intermediates, modelName, LIBND4J_DEFAULT_BASE_DIR, execType);
        }
    }

    public static void checkIntermediate(Map<String, INDArray> inputs, Map<String, INDArray> predictions, Map<String, INDArray[]> intermediates, String modelName, String baseDir, ExecuteWith execType) throws IOException {
        if (!execType.equals(ExecuteWith.SAMEDIFF)) {
            throw new IllegalArgumentException("Currently not supported");
        }
        Nd4j.EPS_THRESHOLD = 1e-4;
        INDArray nd4jPred = null;
        Nd4j.getExecutioner().enableDebugMode(true);
        Nd4j.getExecutioner().enableVerboseMode(true);
        val graph = getGraph(baseDir, modelName);
        if (!inputs.isEmpty()) {
            graph.execWithPlaceHolder(inputs); //This is expected to be just one result
        } else {
            graph.execAndEndResult("output"); //there are tests with no placeholders in the graph like g_00
        }
        for (String varName : graph.variableMap().keySet()) {
            if (!inputs.containsKey(varName)) { //avoiding placeholders
                if (!intermediates.containsKey(varName)) throw new RuntimeException(varName + "not found in resources");
                if (intermediates.get(varName).length > 1)
                    throw new RuntimeException(varName + "has more than one output. Currently not handled");
                assertEquals("Shape not equal on node " + varName, ArrayUtils.toString(intermediates.get(varName)[0].shape()), ArrayUtils.toString(graph.getVariable(varName).getResultShape()));
                //This is acting strange; I can't get it to give proper values - I want to assert equality of values
                //assertEquals(intermediates.get(varName)[0], graph.getArrForVertexId(graph.variableMap().get(varName).getVertexId()))
                log.info("\n\tShapes equal for " + varName);
            }
        }
    }

    public static SameDiff getGraph(String baseDir, String modelName) throws IOException {
        log.info("\n\tRUNNING TEST " + modelName + "...");
        return TFGraphMapper.getInstance().importGraph(new ClassPathResource(baseDir + "/" + modelName + "/frozen_model.pb").getInputStream());
    }

    private static String[] modelDirNames(String base_dir) throws IOException {
        ResourcePatternResolver resolver = new PathMatchingResourcePatternResolver(new ClassPathResource(base_dir).getClassLoader());
        Resource[] resources = resolver.getResources("classpath*:" + base_dir + "/**/frozen_model.pb");
        String[] exampleNames = new String[resources.length];
        for (int i = 0; i < resources.length; i++) {
            exampleNames[i] = resources[i].getURL().toString().split(base_dir + "/")[1].split("/")[0];
        }
        return exampleNames;
    }

    protected static Map<String, INDArray> inputVars(String modelName, String base_dir) throws IOException {
        return readVars(modelName, base_dir, "placeholder");
    }


    protected static Map<String, INDArray> outputVars(String modelName, String base_dir) throws IOException {
        return readVars(modelName, base_dir, "prediction");
    }

    /**
     * Possible for a single node to give multiple outputs though none of the tests cover this currently
     */
    protected static Map<String, INDArray[]> intermediateVars(String modelName, String base_dir) throws IOException {
        Map<String, INDArray> nodeSepOutput = readVars(modelName, base_dir, "prediction_inbw");
        Map<String, Integer> nodeNamesOutputCount = nodeNamesWithOutputCount(nodeSepOutput.keySet());

        Map<String, INDArray[]> nodeWithOutput = new HashMap<>();

        for (String nodeName : nodeNamesOutputCount.keySet()) {
            INDArray[] outputsForNode = new INDArray[nodeNamesOutputCount.get(nodeName)];
            for (int i = 0; i < nodeNamesOutputCount.get(nodeName); i++) {
                outputsForNode[i] = nodeSepOutput.get(nodeName + "." + i); // zero indexed
            }
            nodeWithOutput.put(nodeName, outputsForNode);
        }

        return nodeWithOutput;
    }

    private static Map<String, Integer> nodeNamesWithOutputCount(Set<String> nodesWithOutput) {
        Map<String, Integer> nodeWithOutputCount = new HashMap<>();
        for (String nodeWithOutput : nodesWithOutput) {
            String nodeName = nodeWithOutput.substring(0, nodeWithOutput.lastIndexOf('.'));
            if (nodeWithOutputCount.containsKey(nodeName)) {
                nodeWithOutputCount.put(nodeName, nodeWithOutputCount.get(nodeName) + 1);
            } else {
                nodeWithOutputCount.put(nodeName, 1);
            }
        }
        return nodeWithOutputCount;
    }

    protected static Map<String, INDArray> readVars(String modelName, String base_dir, String pattern) throws IOException {
        Map<String, INDArray> varMap = new HashMap<>();
        String modelDir = base_dir + "/" + modelName;
        ResourcePatternResolver resolver = new PathMatchingResourcePatternResolver(new ClassPathResource(modelDir).getClassLoader());
        Resource[] resources = resolver.getResources("classpath*:" + modelDir + "/**." + pattern + ".shape");
        for (int i = 0; i < resources.length; i++) {
            String fileName = resources[i].getFilename();
            String varPath = modelDir + "/" + fileName;
            String varName = fileName.split("." + pattern + ".shape")[0];
            int[] varShape = Nd4j.readNumpy(new ClassPathResource(varPath).getInputStream(), ",").data().asInt();
            if (varShape.length == 1) {
                if (varShape[0] == 0) {
                    varShape = new int[]{1, 1}; //scalars are mapped to a 1,1 INDArray
                } else {
                    int vectorSize = varShape[0];
                    varShape = new int[]{1, vectorSize}; //vectors are mapped to a row vector
                }
            }
            INDArray varValue = Nd4j.readNumpy(new ClassPathResource(modelDir + "/" + varName + "." + pattern + ".csv").getInputStream(), ",").reshape(varShape);
            if (varName.contains("____")) {
                //these are intermediate node outputs
                varMap.put(varName.replaceAll("____", "/"), varValue);
            } else {
                varMap.put(varName, varValue);
            }
        }
        return varMap;
    }
}
