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
import org.junit.*;
import org.junit.rules.TestWatcher;
import org.junit.runner.Description;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.OpValidationSuite;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Created by susaneraly on 11/29/17.
 */
@Slf4j
@RunWith(Parameterized.class)
public class TFGraphTestAllSameDiff {   //Note: Can't extend BaseNd4jTest here as we need no-arg constructor for parameterized tests

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

    public static final String[] IGNORE_REGEXES = new String[]{
            //Failing 2019/07/01 - Issue 10, https://github.com/deeplearning4j/deeplearning4j/issues/6958
            //Still failing 2019/09/11
            //Still failing 2020/04/27
            //java.lang.IllegalStateException: Requested output variable LogMatrixDeterminant:1 does not exist in SameDiff instance
            "slogdet/.*",

            //Failing 2019/09/11 - https://github.com/eclipse/deeplearning4j/issues/7965
            // Still failing 2020/04/27 java.lang.IllegalStateException: Requested output variable Bincount does not exist in SameDiff instance
            "bincount/.*",
            // Failing 2019/11/14 https://github.com/eclipse/deeplearning4j/issues/8393
            "is_strictly_increasing/emptyArrayTest/.*",

            //TODO floormod and truncatemod behave differently - i.e., "c" vs. "python" semantics. Need to check implementations too
            // Still failing 2020/04/27 java.lang.IllegalStateException: Could not find class for TF Ops: TruncateMod
            "truncatemod/.*",

            //Still failing as of 2019/09/11 - https://github.com/deeplearning4j/deeplearning4j/issues/6464 - not sure if related to: https://github.com/deeplearning4j/deeplearning4j/issues/6447
            "cnn2d_nn/nhwc_b1_k12_s12_d12_SAME",

            //2019/09/11 - No tensorflow op found for SparseTensorDenseAdd
            // 2020/04/27 java.lang.IllegalStateException: Could not find class for TF Ops: SparseTensorDenseAdd
            "confusion/.*",

            //2019/09/11 - Couple of tests failing (InferenceSession issues)
            // Still failing 2020/04/27 Requested output variable concat does not exist in SameDiff instance
            "rnn/bstack/d_.*",

            //2019/05/21 - Failing on AVX2/512 intermittently (Linux, OSX), passing elsewhere
            //"unsorted_segment/.*",

            //2019/05/21 - Failing on windows-x86_64-cuda-9.2 only -
            "conv_4",
            "g_09",
            //"unsorted_segment/unsorted_segment_mean_rank2",

            //2019/05/28 - JVM crash on ppc64le only - See issue 7657
            "g_11",

            //2019/07/09 - Need "Multinomial" op - https://github.com/eclipse/deeplearning4j/issues/7913
            // Still failing 2020/04/27 java.lang.IllegalStateException: Could not find class for TF Ops: Multinomial
            "multinomial/.*",

            //2019/11/04 AB - disabled, pending libnd4j deconv3d_tf implementation
            // Still failing 2020/04/27 java.lang.IllegalStateException: Could not find descriptor for op: deconv3d_tf - class: org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv3DTF
            "conv3d_transpose.*",

            //2019/11/15 - mapping is not present yet https://github.com/eclipse/deeplearning4j/issues/8397
            // Still failing 2020/04/27 java.lang.AssertionError: Predictions do not match on ragged/reduce_mean/2d_a1, node RaggedReduceMean/truediv
            "ragged/reduce_mean/.*",

            // 2019/11/15 - missing dtype argument in nd4j, tests are useless https://github.com/eclipse/deeplearning4j/issues/8398
            // Still failing 2020/04/27 java.lang.IndexOutOfBoundsException: 1
           "zeros_like/rank2_float32_dtype_int.*",

            // 11.26.2019 failing - https://github.com/eclipse/deeplearning4j/issues/8453
            // Still failing 2020/04/27 java.lang.AssertionError: Predictions do not match on roll/rank2_float32_zeroshift, node Roll
            "roll/.*",

            // 11.26.2019 failing https://github.com/eclipse/deeplearning4j/issues/8455
            // still failing 2020/04/27
            // java.lang.IllegalStateException: Failed to calculate output shapes for op matrix_band_part (MatrixBandPart) - no shapes were returned by calculateOutputShape()
            "matrix_band_part/.*",

            // 12.20.2019 - https://github.com/eclipse/deeplearning4j/issues/8559
            // Still failing 2020/27/04 java.lang.AssertionError: Predictions do not match on fused_batch_norm/float32_nhcw, node FusedBatchNormV3
            "fused_batch_norm/.*",

            // 01.05.2020 -  https://github.com/eclipse/deeplearning4j/issues/8898
             "primitive_gru",

            // 05.05.2020 - https://github.com/eclipse/deeplearning4j/issues/8921
            "random_poisson/rank1_float16", "random_poisson/rank1_float32", "random_poisson/rank1_float16", "random_poisson/rank1_half",
            "random_poisson_v2/rank1_float64", "random_poisson_v2/rank1_float16", "random_poisson_v2/rank1_half",

            //08.05.2020 - https://github.com/eclipse/deeplearning4j/issues/8927
            "random_gamma/.*",

            //08.05.2020 - https://github.com/eclipse/deeplearning4j/issues/8928
            "Conv3DBackpropInputV2/.*",

            //12.05.2020 - https://github.com/eclipse/deeplearning4j/issues/8940
            "compare_and_bitpack/.*",

            //12.05.2020 - https://github.com/eclipse/deeplearning4j/issues/8943
            "max_pool_with_argmax/int64_int64_padding_SAME", "max_pool_with_argmax/int32_int64_padding_SAME",

            //12.05.2020 - https://github.com/eclipse/deeplearning4j/issues/8946
            "non_max_suppression_v4/.*","non_max_suppression_v5/.*",

            // 18.05.2020 - https://github.com/eclipse/deeplearning4j/issues/8960
            "random_shuffle/.*",
            // 18.05.2020 - https://github.com/eclipse/deeplearning4j/issues/8963
            "random_uniform/.*",
            "random_uniform_int/.*",
            "random_normal/.*",
            "random_gamma/.*",
            "random_poisson/.*",
            "random_poisson/.*",
            "random_poisson_v2/.*",

  };

    /* As per TFGraphTestList.printArraysDebugging - this field defines a set of regexes for test cases that should have
       all arrays printed during execution.
       If a test name matches any regex here, an ExecPrintListener will be added to the listeners, and all output
       arrays will be printed during execution
     */
    private final List<String> debugModeRegexes = null; //Arrays.asList("resize_nearest_neighbor/.*", "add_n.*");

    @BeforeClass
    public static void beforeClass() {
        Nd4j.setDataType(DataType.FLOAT);
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
    }

    @Before
    public void setup() {
        Nd4j.setDataType(DataType.FLOAT);
        Nd4j.getExecutioner().enableDebugMode(false);
        Nd4j.getExecutioner().enableVerboseMode(false);
    }

    @After
    public void tearDown() {
    }

    @Parameterized.Parameters(name="{2}")
    public static Collection<Object[]> data() throws IOException {
        val localPath = System.getenv(TFGraphTestAllHelper.resourceFolderVar);

        // if this variable isn't set - we're using dl4j-tests-resources
        if (localPath == null) {
            File baseDir = new File(System.getProperty("java.io.tmpdir"), UUID.randomUUID().toString());
            List<Object[]> params = TFGraphTestAllHelper.fetchTestParams(BASE_DIR, MODEL_FILENAME, EXECUTE_WITH, baseDir);
            return params;
        } else {
            File baseDir = new File(localPath);
            return TFGraphTestAllHelper.fetchTestParams(BASE_DIR, MODEL_FILENAME, EXECUTE_WITH, baseDir);
        }
    }

    public TFGraphTestAllSameDiff(Map<String, INDArray> inputs, Map<String, INDArray> predictions, String modelName, File localTestDir) {
        this.inputs = inputs;
        this.predictions = predictions;
        this.modelName = modelName;
        this.localTestDir = localTestDir;
    }

    @Test//(timeout = 25000L)
    public void testOutputOnly() throws Exception {
        if(TFGraphTestZooModels.isPPC()){
            /*
            Ugly hack to temporarily disable tests on PPC only on CI
            Issue logged here: https://github.com/deeplearning4j/deeplearning4j/issues/7657
            These will be re-enabled for PPC once fixed - in the mean time, remaining tests will be used to detect and prevent regressions
             */

            log.warn("TEMPORARILY SKIPPING TEST ON PPC ARCHITECTURE DUE TO KNOWN JVM CRASH ISSUES - SEE https://github.com/deeplearning4j/deeplearning4j/issues/7657");
            OpValidationSuite.ignoreFailing();
        }


        Nd4j.create(1);

        for(String s : IGNORE_REGEXES){
            if(modelName.matches(s)){
                log.info("\n\tIGNORE MODEL ON REGEX: {} - regex {}", modelName, s);
                OpValidationSuite.ignoreFailing();
            }
        }
        Pair<Double,Double> precisionOverride = TFGraphTestAllHelper.testPrecisionOverride(modelName);
        Double maxRE = (precisionOverride == null ? null : precisionOverride.getFirst());
        Double minAbs = (precisionOverride == null ? null : precisionOverride.getSecond());

        boolean verboseDebugMode = false;
        if(debugModeRegexes != null){
            for(String regex : debugModeRegexes){
                if(modelName.matches(regex)){
                    verboseDebugMode = true;
                    break;
                }
            }
        }

        try {
            TFGraphTestAllHelper.checkOnlyOutput(inputs, predictions, modelName, BASE_DIR, MODEL_FILENAME, EXECUTE_WITH, TFGraphTestAllHelper.LOADER, maxRE, minAbs, verboseDebugMode);
            //TFGraphTestAllHelper.checkIntermediate(inputs, modelName, BASE_DIR, MODEL_FILENAME, EXECUTE_WITH, localTestDir);
        } catch (Throwable t){
            log.error("ERROR Executing test: {} - input keys {}", modelName, (inputs == null ? null : inputs.keySet()), t);
            throw t;
        }
        //TFGraphTestAllHelper.checkIntermediate(inputs, modelName, EXECUTE_WITH);
    }

}
