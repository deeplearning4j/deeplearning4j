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
 * TFGraphTestAll* will run all the checked in TF graphs and
 * compare outputs in nd4j to those generated and checked in from TF.
 * <p>
 * This file is to run a single graph or a list of graphs to aid in debug.
 * Simply change the modelNames String[] to correspond to the directory name the graph lives in
 * - eg. to run the graph for 'bias_add' i.e checked in under tf_graphs/examples/bias_add
 * <p>
 *
 */
@RunWith(Parameterized.class)
public class TFGraphTestList {

    public static String[] modelNames = new String[]{
//            "add_n",
            //"ae",
            //"ae_00",
            "bias_add",
            // "concat"
            // "conv_0",
            // "conv_1",
            // "conv_2",
            // "conv_3",
            //"deep_mnist", //NOTE THIS ONE WILL FAIL because it is expecting a placeholder value for dropout % which we tie to 1.0 in inference
            //"deep_mnist_no_dropout", //Takes way too long since there are a lot of nodes, would skip for now
            // "expand_dim",

            // "g_00", //This has no placeholders in the graph - not sure how to exec as it gives a NPE
            //   "g_01",
            //   "g_01",
               //"g_02",
            //   "g_03",
           //   "g_04",
             //  "g_05",
           //   "gru_mnist",
            //"lstm_mnist",
            // "math_mul_order",
            //"mlp_00",
            //"mnist_00",
            //  "node_multiple_out",// -> Need to map multiple out values to graph node output values
            //  "norm_tests",
            //  "pool_0",
           //   "pool_1",
            //  "primitive_gru",
           //   "primitive_gru_dynamic",
             // "primitive_lstm",
            //  "stack",
            //  "stack_1d",
            //"stack_scalar",
          //   "simple_cond"
            //"transform_0",
            //"transpose",
            //"transpose_00",
             // "unstack"
    };

    /**
     * Failures:
     * transpose
     * simple_cond
     * primitive_gru_dynamic (while loop related)
     * pool_1 (attributes related)
     * pool_0: (attributes related)
     * norm_tests: invalid test (missing resources)
     * g_05: Gather related
     * g_04: assertion failure
     */

    //change this to SAMEDIFF for samediff
    public static TFGraphTestAllHelper.ExecuteWith executeWith = ExecuteWith.SAMEDIFF;
    //public static TFGraphTestAllHelper.ExecuteWith executeWith = TFGraphTestAllHelper.ExecuteWith.LIBND4J;
    //public static TFGraphTestAllHelper.ExecuteWith executeWith = TFGraphTestAllHelper.ExecuteWith.JUST_PRINT;

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
    public void testOutputOnly() throws IOException {
        Map<String, INDArray> inputs = inputVars(modelName, modelDir);
        Map<String, INDArray> predictions = outputVars(modelName, modelDir);
        checkOnlyOutput(inputs, predictions, modelName, modelDir, executeWith);
    }

    @Test
    public void testAlsoIntermediate() throws IOException {
        Map<String, INDArray> inputs = inputVars(modelName, modelDir);
        checkIntermediate(inputs, modelName, executeWith);

    }
}
