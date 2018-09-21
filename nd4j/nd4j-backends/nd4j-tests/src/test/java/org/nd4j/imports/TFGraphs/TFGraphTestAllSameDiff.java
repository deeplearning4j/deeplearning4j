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
import org.junit.*;
import org.junit.rules.TestWatcher;
import org.junit.runner.Description;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.OpValidationSuite;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Created by susaneraly on 11/29/17.
 */
@Slf4j
@RunWith(Parameterized.class)
public class TFGraphTestAllSameDiff {

    @Rule
    public TestWatcher testWatcher = new TestWatcher() {

        @Override
        protected void starting(Description description){
            log.info("TFGraphTestAllSameDiff: Starting parameterized test: " + description.getDisplayName());
        }

        //protected void failed(Throwable e, Description description) {
        //protected void succeeded(Description description) {
    };

    private Map<String, INDArray> inputs;
    private Map<String, INDArray> predictions;
    private String modelName;
    private File localTestDir;

    private static final TFGraphTestAllHelper.ExecuteWith EXECUTE_WITH = TFGraphTestAllHelper.ExecuteWith.SAMEDIFF;
    private static final String BASE_DIR = "tf_graphs/examples";
    private static final String MODEL_FILENAME = "frozen_model.pb";



    private static final String[] SKIP_ARR = new String[] {
            "deep_mnist",
            "deep_mnist_no_dropout",
            "ssd_mobilenet_v1_coco",
            "yolov2_608x608",
            "inception_v3_with_softmax",
            "conv_5", // this test runs, but we can't make it pass atm due to different RNG algorithms
    };

    public static final String[] IGNORE_REGEXES = new String[]{
            //https://github.com/deeplearning4j/deeplearning4j/issues/6154
            "transforms/atan2_3,1,4_1,2,4",
            //https://github.com/deeplearning4j/deeplearning4j/issues/6142
            "reverse/shape5.*",
            //https://github.com/deeplearning4j/deeplearning4j/issues/6172
            "pad/rank1.*",
            "pad/rank2Pone_const10",
            "pad/rank3.*",
            //https://github.com/deeplearning4j/deeplearning4j/issues/6177
            "topk/.*",
            //https://github.com/deeplearning4j/deeplearning4j/issues/6179
            "in_top_k/.*",
            //https://github.com/deeplearning4j/deeplearning4j/issues/6181
            "confusion/.*",
            //https://github.com/deeplearning4j/deeplearning4j/issues/6180
            "identity_n.*",
            //https://github.com/deeplearning4j/deeplearning4j/issues/6182
            "zeta.*",

            //https://github.com/deeplearning4j/deeplearning4j/issues/6281
            "log_determinant/.*",
            "slogdet/.*",

            //https://github.com/deeplearning4j/deeplearning4j/issues/6285
            "histogram_fixed_width/.*",

            //TODO need unsorted segment sum - then need to change libnd4j impl slightly (need to know format first)
            "bincount/.*",

            //Crashing?
            "batchnorm/.*",

            //Not sure what's up here - "DEPTHWISECONV2D OP: wrong shape of weights array, expected is [-1, -1, 2, 2], but got [1, 2, 2, 2] instead !"
            "sepconv1d_layers/.*",

            //scatter_nd: one minor validation issue mentioned tu Yurii, already fixed but not merged (should validate vs. shape array length, not rank)
            "scatter_nd/.*",

            //https://github.com/deeplearning4j/deeplearning4j/issues/6421
            "conv2d_transpose/.*",

            //https://github.com/deeplearning4j/deeplearning4j/issues/6311
            "embedding_lookup/.*",

            //https://github.com/deeplearning4j/deeplearning4j/issues/6315
            "nth_element/.*",

            //https://github.com/deeplearning4j/deeplearning4j/issues/6290
            "unsorted_segment/.*",

            //https://github.com/deeplearning4j/deeplearning4j/issues/6321
            "broadcast_to/.*",

            //https://github.com/deeplearning4j/deeplearning4j/issues/6322
            "broadcast_dynamic_shape/.*",

            //https://github.com/deeplearning4j/deeplearning4j/issues/6345
            "where/cond_only_rank.*",

            //https://github.com/deeplearning4j/deeplearning4j/issues/6346
            "boolean_mask/.*",

            //TODO floormod and truncatemod behave differently - i.e., "c" vs. "python" semantics. Need to check implementations too
            "truncatemod/.*",

            //Not sure why these are failing yet
            "lrn/dr3.*",
            "lrn/dr5.*",

            //This is failing on strided slice - on what appears to be an invalid op arguments. Will replace
            // this test with a set of more thorough/isolated strided slice tests
            "g_07",

            //https://github.com/deeplearning4j/deeplearning4j/issues/6447
            "cnn1d_layers/channels_first_b2_k2_s1_d2_SAME",
            "cnn2d_layers/channels_first_b1_k12_s1_d12_SAME",

            //https://github.com/deeplearning4j/deeplearning4j/issues/6441
            "matrix_determinant/rank2_5,5",

            //These have a random component so can't be validated using simple .equals... should still be compared, however
            "alpha_dropout/.*",
            "layers_dropout/.*",

            //These absurdly slow:
            "simplewhile.*",

            //New failures:
            "gru_dynamic_mnist",
            "primitive_gru_dynamic",
            "simple_while",

            //Bad test, no outputs (but there are non-output ("inbetween") results)
            "g_10",

            //https://github.com/deeplearning4j/deeplearning4j/issues/6464
            "cnn2d_nn/nchw_b1_k12_s12_d12_SAME",
            "cnn2d_nn/nhwc_b1_k12_s12_d12_SAME",

            "conv_4"
    };
    public static final Set<String> SKIP_SET = new HashSet<>(Arrays.asList(SKIP_ARR));

    @BeforeClass
    public static void beforeClass() throws Exception {
        Nd4j.setDataType(DataBuffer.Type.FLOAT);
    }

    @Before
    public void setup() {
        Nd4j.setDataType(DataBuffer.Type.FLOAT);
    }

    @After
    public void tearDown() throws Exception {
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(true);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(true);
    }

    @Parameterized.Parameters(name="{2}")
    public static Collection<Object[]> data() throws IOException {
        File baseDir = new File(System.getProperty("java.io.tmpdir"), UUID.randomUUID().toString());
        List<Object[]> params = TFGraphTestAllHelper.fetchTestParams(BASE_DIR, MODEL_FILENAME, EXECUTE_WITH, baseDir);
        return params;
    }

    public TFGraphTestAllSameDiff(Map<String, INDArray> inputs, Map<String, INDArray> predictions, String modelName, File localTestDir) throws IOException {
        this.inputs = inputs;
        this.predictions = predictions;
        this.modelName = modelName;
        this.localTestDir = localTestDir;
    }

    @Test(timeout = 25000L)
    public void testOutputOnly() throws Exception {
        Nd4j.create(1);
        if (SKIP_SET.contains(modelName)) {
            log.info("\n\tSKIPPED MODEL: " + modelName);
            return;
        }

        for(String s : IGNORE_REGEXES){
            if(modelName.matches(s)){
                log.info("\n\tIGNORE MODEL ON REGEX: {} - regex {}", modelName, s);
                OpValidationSuite.ignoreFailing();
            }
        }
        Pair<Double,Double> precisionOverride = TFGraphTestAllHelper.testPrecisionOverride(modelName);
        Double maxRE = (precisionOverride == null ? null : precisionOverride.getFirst());
        Double minAbs = (precisionOverride == null ? null : precisionOverride.getSecond());

        TFGraphTestAllHelper.checkOnlyOutput(inputs, predictions, modelName, BASE_DIR, MODEL_FILENAME, EXECUTE_WITH,
                TFGraphTestAllHelper.LOADER, maxRE, minAbs);
        //TFGraphTestAllHelper.checkIntermediate(inputs, modelName, EXECUTE_WITH);
    }

}
