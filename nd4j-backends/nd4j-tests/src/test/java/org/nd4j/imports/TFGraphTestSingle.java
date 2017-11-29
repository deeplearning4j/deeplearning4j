package org.nd4j.imports;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

import static org.nd4j.imports.TFGraphTestAll.*;

/**
 * TFGraphTestAll will run all the checked in TF graphs and
 *      compare outputs in nd4j to those generated and checked in from TF.
 *
 * This file is to run a single graph that is checked in to aid in debug.
 * Simply change the modelName String in testOne() to correspond to the directory name the graph lives in
 *  - eg. to run the graph for 'bias_add' i.e checked in under tf_graphs/examples/bias_add
 *  set modelName to "bias_add"
 *
 */
@Slf4j
public class TFGraphTestSingle {

    @Test
    public void testOne() throws  Exception {
        //String modelName = "add_n";
        //String modelName = "ae_00";
        String modelName = "bias_add";
        //String modelName = "conv_0";
        //String modelName = "g_00";
        //String modelName = "g_01";
        //String modelName = "math_mul_order";
        //String modelName = "mlp_00";
        //String modelName = "mlp_00_test";
        //String modelName = "transform_0";
        //String modelName = "transpose";
        Map<String, INDArray> inputs = inputVars(modelName);
        Map<String, INDArray> predictions = outputVars(modelName);
        testSingle(inputs,predictions,modelName);
    }
}
