package org.nd4j.imports;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
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
public class TFGraphTestAll {

    private Map<String, INDArray> inputs;
    private Map<String, INDArray> predictions;
    private String modelName;
    private static final String BASE_DIR = "tf_graphs/examples";

    @Parameterized.Parameters
    public static Collection<Object[]> data() throws IOException {
        String[] modelNames = modelDirNames();
        List<Object[]> modelParams = new ArrayList<>();
        for (int i = 0; i < modelNames.length; i++) {
            Object[] currentParams = new Object[3];
            currentParams[0] = inputVars(modelNames[i]); //input variable map - could be null
            currentParams[1] = outputVars(modelNames[i]); //saved off predictions
            currentParams[2] = modelNames[i];
            modelParams.add(currentParams);
        }
        return modelParams;
    }

    public TFGraphTestAll(Map<String, INDArray> inputs, Map<String, INDArray> predictions, String modelName) throws IOException {
        this.inputs = inputs;
        this.predictions = predictions;
        this.modelName = modelName;
    }

    //Missing bias add fix currently
    @Test
    public void test() throws Exception {
        Nd4j.create(1);
        testSingle(inputs, predictions, modelName);
    }

    protected static void testSingle(Map<String, INDArray> inputs, Map<String, INDArray> predictions, String modelName) throws IOException {
        Nd4j.EPS_THRESHOLD = 1e-4;
        log.info("\n\tRUNNING TEST " + modelName + "...");
        val graph = TFGraphMapper.getInstance().importGraph(new ClassPathResource(BASE_DIR + "/" + modelName + "/frozen_model.pb").getInputStream());
        INDArray res = graph.execWithPlaceHolderAndEndResult(inputs);

        //for (int i = 0; i < res.length; i++) {
        //    if (i > 0)
        //throw new IllegalArgumentException("NOT CURRENTLY SUPPORTED BY WORKFLOW"); //figure out how to support multiple outputs with freezing in TF
        //INDArray nd4jPred = res[i];
        INDArray nd4jPred = res;
        INDArray tfPred = predictions.get("output");
        assertEquals("Predictions do not match on " + modelName, tfPred.reshape(nd4jPred.shape()), nd4jPred);
            /*
            try {
                assertTrue(Transforms.abs(tfPred.reshape(nd4jPred.shape()).sub(nd4jPred)).maxNumber().floatValue() < 1e-8);
            } catch (Throwable t) {
                assertEquals("Predictions do not match on " + modelName, tfPred.reshape(nd4jPred.shape()), nd4jPred);
            }
            */
        //}
        log.info("\n\tTEST " + modelName + " PASSED...");
        log.info("\n========================================================\n");

    }

    protected static String[] modelDirNames() throws IOException {
        ResourcePatternResolver resolver = new PathMatchingResourcePatternResolver(new ClassPathResource(BASE_DIR).getClassLoader());
        Resource[] resources = resolver.getResources("classpath*:"+ BASE_DIR + "/**/frozen_model.pb");
        String[] exampleNames = new String[resources.length];
        for (int i =0 ; i < resources.length; i++) {
            exampleNames[i] = resources[i].getURL().toString().split(BASE_DIR+"/")[1].split("/")[0];
        }
        return exampleNames;
    }

    protected static Map<String, INDArray> inputVars(String modelName) throws IOException {
        Map<String, INDArray> inputVarMap = new HashMap<>();
        String modelDir = BASE_DIR + "/" + modelName;
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
    protected static Map<String, INDArray> outputVars(String modelName) throws IOException {
        Map<String, INDArray> outputVarMap = new HashMap<>();
        String modelDir = BASE_DIR + "/" + modelName;
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
