package org.nd4j.imports;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.autodiff.execution.NativeGraphExecutioner;
import org.nd4j.autodiff.execution.conf.ExecutionMode;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.core.io.support.ResourcePatternResolver;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * Created by susaneraly on 11/6/17.
 */
@Slf4j
@RunWith(Parameterized.class)
public class TFGraphTestAllHelper {

    public  enum ExecuteWith {
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
            Object[] currentParams = new Object[3];
            currentParams[0] = inputVars(modelNames[i], baseDir); //input variable map - could be null
            currentParams[1] = outputVars(modelNames[i], baseDir); //saved off predictions
            currentParams[2] = modelNames[i];
            modelParams.add(currentParams);
        }
        return modelParams;
    }

    protected static List<Object[]> fetchTestParams(String baseDir, String modelName) throws IOException {
        List<Object[]> modelParams = new ArrayList<>();
        Object[] currentParams = new Object[3];
        currentParams[0] = inputVars(modelName, baseDir); //input variable map - could be null
        currentParams[1] = outputVars(modelName, baseDir); //saved off predictions
        currentParams[2] = modelName;
        modelParams.add(currentParams);
        return modelParams;
    }

    protected static void testSingle(Map<String, INDArray> inputs, Map<String, INDArray> predictions, String modelName, ExecuteWith execType) throws IOException {
        if (execType.equals(ExecuteWith.SAMEDIFF)) {
            testSingle(inputs, predictions, modelName, SAMEDIFF_DEFAULT_BASE_DIR, execType);
        } else if (execType.equals(ExecuteWith.LIBND4J)) {
            testSingle(inputs, predictions, modelName, LIBND4J_DEFAULT_BASE_DIR, execType);
        }
    }

    protected static void testSingle(Map<String, INDArray> inputs, Map<String, INDArray> predictions, String modelName, String baseDir, ExecuteWith execType) throws IOException {
        Nd4j.EPS_THRESHOLD = 1e-4;
        log.info("\n\tRUNNING TEST " + modelName + "...");
        val graph = TFGraphMapper.getInstance().importGraph(new ClassPathResource(baseDir + "/" + modelName + "/frozen_model.pb").getInputStream());
        INDArray nd4jPred = null;
        Nd4j.getExecutioner().enableDebugMode(true);
        Nd4j.getExecutioner().enableVerboseMode(true);

        if (execType.equals(ExecuteWith.SAMEDIFF)) {
            graph.execWithPlaceHolder(inputs); //This is expected to be just one result
            nd4jPred = graph.getVariable("output").getArr();
        } else if (execType.equals(ExecuteWith.LIBND4J)) {
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
        Map<String, INDArray> inputVarMap = new HashMap<>();
        String modelDir = base_dir + "/" + modelName;
        ResourcePatternResolver resolver = new PathMatchingResourcePatternResolver(new ClassPathResource(modelDir).getClassLoader());
        Resource[] resources = resolver.getResources("classpath*:" + modelDir + "/**.shape");
        for (int i = 0; i < resources.length; i++) {
            String inputFileName = resources[i].getFilename();
            String inputPath = modelDir + "/" + inputFileName;
            String inputName = inputFileName.split(".shape")[0];
            int[] inputShape = Nd4j.readNumpy(new ClassPathResource(inputPath).getInputStream(), ",").data().asInt();
            INDArray input = Nd4j.readNumpy(new ClassPathResource(modelDir + "/" + inputName + ".csv").getInputStream(), ",").reshape(inputShape);
            inputVarMap.put(inputName, input);
        }
        return inputVarMap;
    }


    //TODO: I don't check shapes
    protected static Map<String, INDArray> outputVars(String modelName, String base_dir) throws IOException {
        Map<String, INDArray> outputVarMap = new HashMap<>();
        String modelDir = base_dir + "/" + modelName;
        ResourcePatternResolver resolver = new PathMatchingResourcePatternResolver(new ClassPathResource(modelDir).getClassLoader());
        Resource[] resources = resolver.getResources("classpath*:" + modelDir + "/**prediction.csv");
        for (int i = 0; i < resources.length; i++) {
            String outputFileName = resources[i].getFilename();
            String outputPath = modelDir + "/" + outputFileName;
            String outputName = outputFileName.split(".prediction.csv")[0];
            INDArray output = Nd4j.readNumpy(new ClassPathResource(outputPath).getInputStream(), ",");
            outputVarMap.put(outputName, output);
        }
        return outputVarMap;
    }
}
