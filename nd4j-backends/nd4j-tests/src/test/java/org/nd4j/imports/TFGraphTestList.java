package org.nd4j.imports;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import static org.nd4j.imports.TFGraphTestAllHelper.*;

/**
 * TFGraphTestAll will run all the checked in TF graphs and
 * compare outputs in nd4j to those generated and checked in from TF.
 * <p>
 * This file is to run a single graph or a list of graphs that are checked in to aid in debug.
 * Simply change the modelNames String[] in testSome() to correspond to the directory name the graph lives in
 * - eg. to run the graph for 'bias_add' i.e checked in under tf_graphs/examples/bias_add
 * set modelNames to "bias_add"
 */
@RunWith(Parameterized.class)
public class TFGraphTestList {

    public static String[] modelNames = new String[]{
         //   "add_n",
           //"ae_00",
           // "bias_add",
           // "conv_0",
            //"deep_mnist",
            //NOTE THIS ONE IS BROKEN: THE INPUT VARIABLE IS NOT PRESENT
           // "deep_mnist_no_dropout",
            //"g_00",
            //"g_01",
            //"math_mul_order",
            //"mlp_00",
            //"mnist_00",
            //"transform_0",
            //"transpose",
    };
    //change this to SAMEDIFF for samediff
    public static TFGraphTestAllHelper.ExecuteWith executeWith = TFGraphTestAllHelper.ExecuteWith.SAMEDIFF;

   // public static TFGraphTestAllHelper.ExecuteWith executeWith = TFGraphTestAllHelper.ExecuteWith.LIBND4J;
    public static String modelDir = TFGraphTestAllHelper.COMMON_BASE_DIR; //this is for later if we want to check in models separately for samediff and libnd4j

    private String modelName;

    @Parameterized.Parameters
    public static Collection<Object[]> data() throws IOException {
        List<Object[]> modelNamesParams = new ArrayList<>();
        for (int i = 0; i < modelNames.length; i++) {
            Object[] currentParams = new String[]{modelNames[i]};
            modelNamesParams.add(currentParams);
        }
        return modelNamesParams;
    }

    public TFGraphTestList(String modelName) {
        this.modelName = modelName;
    }

    @Test
    public void testSome() throws IOException {
        Map<String, INDArray> inputs = inputVars(modelName, modelDir);
        Map<String, INDArray> predictions = outputVars(modelName, modelDir);
        testSingle(inputs, predictions, modelName, modelDir, executeWith);
    }
}
