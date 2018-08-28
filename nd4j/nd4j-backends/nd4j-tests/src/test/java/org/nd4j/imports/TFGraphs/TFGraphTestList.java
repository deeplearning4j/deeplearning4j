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

import org.junit.After;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * TFGraphTestAll* will run all the checked in TF graphs and
 * compare outputs in nd4j to those generated and checked in from TF.
 * <p>
 * This file is to run a single graph or a list of graphs to aid in debug.
 * Simply change the modelNames String[] to correspond to the directory name the graph lives in
 * - eg. to run the graph for 'bias_add' i.e checked in under tf_graphs/examples/bias_add
 * <p>
 */
@RunWith(Parameterized.class)
@Ignore
public class TFGraphTestList {
    public static String[] modelNames = new String[]{
//            "bincount/rank0",
//            "bincount/rank0_minmax",
//            "bincount/rank1_minmax",
//            "bincount/rank1_min10",
//            "bincount/rank1_max5",
//            "bincount/rank1_minmax_weights",
//            "scatter_nd/rank2shape_2indices",
            /*"add_n",
            "ae",
            "ae_00",
            "bias_add",
            "norm_tests/norm_0",
            "concat",
            "conv_0",
            "conv_1", //Raver is working on this
            "conv_2", //missing SpaceToBatchND
            "conv_3", //fails due to 4d input: this seems to be related to Conv2d being mapped to Dilation2D which takes 3d input
           // "deep_mnist", //broadcast bug? double check with raver
          //  "deep_mnist_no_dropout",
            "expand_dim",
            "g_00",
            "g_01",
            "g_01",
            "g_02",
            "g_03", //op missing?
            "g_04",
            "g_05",
            "gru_mnist",
            "lstm_mnist",
            "math_mul_order",
            "mlp_00",
            "mnist_00",
            //"node_multiple_out",
            "non2d_0",
            "non2d_0A",
            "pool_0",
            "pool_1",
            "primitive_gru",
            "primitive_gru_dynamic", //while loop related NullPointer, double check import here
            "primitive_lstm",
          "ssd_mobilenet_v1_coco",
            "stack",
            "stack_1d",
            "stack_scalar",
            "simpleif_0",
            "simple_cond", //JVM crash
            "simple_while",  //Functions not being added: Need to finish while import
            "transform_0",
            "transpose_00",
            "unstack",
            //"yolov2_608x608"*/

    };

    @After
    public void tearDown() throws Exception {
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(false);
    }

    //change this to SAMEDIFF for samediff
    public static TFGraphTestAllHelper.ExecuteWith executeWith = TFGraphTestAllHelper.ExecuteWith.SAMEDIFF;
//    public static TFGraphTestAllHelper.ExecuteWith executeWith = TFGraphTestAllHelper.ExecuteWith.LIBND4J;
    // public static TFGraphTestAllHelper.ExecuteWith executeWith = TFGraphTestAllHelper.ExecuteWith.JUST_PRINT;

    public static String modelDir = TFGraphTestAllHelper.COMMON_BASE_DIR; //this is for later if we want to check in models separately for samediff and libnd4j

    private String modelName;

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
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
        Map<String, INDArray> inputs = TFGraphTestAllHelper.inputVars(modelName, modelDir);
        Map<String, INDArray> predictions = TFGraphTestAllHelper.outputVars(modelName, modelDir);
        Double precisionOverride = TFGraphTestAllHelper.testPrecisionOverride(modelName);
        TFGraphTestAllHelper.checkOnlyOutput(inputs, predictions, modelName, modelDir, executeWith, precisionOverride);
    }

    @Test
    public void testAlsoIntermediate() throws IOException {
        Map<String, INDArray> inputs = TFGraphTestAllHelper.inputVars(modelName, modelDir);
        TFGraphTestAllHelper.checkIntermediate(inputs, modelName, executeWith);

    }
}
