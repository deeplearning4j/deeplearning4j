/*
 *
 *  *  ******************************************************************************
 *  *  *
 *  *  *
 *  *  * This program and the accompanying materials are made available under the
 *  *  * terms of the Apache License, Version 2.0 which is available at
 *  *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *  *
 *  *  *  See the NOTICE file distributed with this work for additional
 *  *  *  information regarding copyright ownership.
 *  *  * Unless required by applicable law or agreed to in writing, software
 *  *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  * License for the specific language governing permissions and limitations
 *  *  * under the License.
 *  *  *
 *  *  * SPDX-License-Identifier: Apache-2.0
 *  *  *****************************************************************************
 *
 *
 */

package org.nd4j.imports.tfgraphs;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.extension.ConditionEvaluationResult;
import org.junit.jupiter.api.extension.ExecutionCondition;
import org.junit.jupiter.api.extension.ExtensionContext;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assumptions.assumeFalse;

@Slf4j
public class TFImportDisableModelsCondition implements ExecutionCondition {

    /**
     * NOTE: If this is empty or the tests names are wrong,
     * all tests will trigger an assumeFalse(..) that indicates
     * the status of the test failing. No tests will run.
     */
    public final static List<String> EXECUTE_ONLY_MODELS = Arrays.asList(
            /*"layers_dropout/rank2_d01_train",
            "layers_dropout/rank4_d05_train",
            "layers_dropout/rank3_d05_train_mask2",
            "layers_dropout/rank4_d05_train_mask",
            "layers_dropout/rank3_d05_train_mask1",
            "layers_dropout/rank2_d09_train",
            "layers_dropout/rank2_d05_train",*/


    );

    public static final String[] IGNORE_REGEXES = new String[]{
            //Failing 2019/09/11 - https://github.com/eclipse/deeplearning4j/issues/7965
            // Still failing 2020/04/27 java.lang.IllegalStateException: Requested output variable Bincount does not exist in SameDiff instance
            //Invalid test cases. Verified by running graph against actual TF.
            "scatter_nd_sub/locking/rank1shape_1indices",
            "reductions/scatter_update_vector",
            "reductions/scatter_update_scalar",
            "emptyArrayTests/scatter_update/rank1_emptyIndices_emptyUpdates",
            "bincount/rank2_weights",
            "slogdet/.*",
            "fused_batch_norm/float16_nhwc",
            "emptyArrayTests/scatter_update/rank2_emptyIndices_emptyUpdates",
            //Don't bother to test RNG. We can test subsets of ops with dropout to make sure they are consistent
            //These tests have random uniform and other RNG in them that don't need to be perfectly compatible to be acceptable.
            //We need different test cases here.
            "layers_dropout/.*",
            //TODO floormod and truncatemod behave differently - i.e., "c" vs. "python" semantics. Need to check implementations too
            // Still failing 2020/04/27 java.lang.IllegalStateException: Could not find class for TF Ops: TruncateMod
            "truncatemod/.*",

            //2019/09/11 - No tensorflow op found for SparseTensorDenseAdd
            // 2020/04/27 java.lang.IllegalStateException: Could not find class for TF Ops: SparseTensorDenseAdd
            "confusion/.*",

            //2019/09/11 - Couple of tests failing (InferenceSession issues)
            // Still failing 2020/04/27 Requested output variable concat does not exist in SameDiff instance


            //2019/05/21 - Failing on windows-x86_64-cuda-9.2 only -
            "conv_4",
            "g_09",

            //2019/05/28 - JVM crash on ppc64le only - See issue 7657
            "g_11",

            //2019/07/09 - Need "Multinomial" op - https://github.com/eclipse/deeplearning4j/issues/7913
            // Still failing 2020/04/27 java.lang.IllegalStateException: Could not find class for TF Ops: Multinomial
            "multinomial/.*",

            //2019/11/04 AB - disabled, pending libnd4j deconv3d_tf implementation
            // Still failing 2020/04/27 java.lang.IllegalStateException: Could not find descriptor for op: deconv3d_tf - class: org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv3DTF
            "conv3d_transpose.*",

            //2019/11/15 - mapping is not present yet https://github.com/eclipse/deepleRaggedRange arning4j/issues/8397
            // Still failing 2020/04/27 java.lang.AssertionError: Predictions do not match on ragged/reduce_mean/2d_a1, node RaggedReduceMean/truediv
            "ragged/reduce_mean/.*",


            //08.05.2020 - https://github.com/eclipse/deeplearning4j/issues/8927
            "random_gamma/.*",

            //08.05.2020 - https://github.com/eclipse/deeplearning4j/issues/8928
            "Conv3DBackpropInputV2/.*",



            //12.05.2020 - https://github.com/eclipse/deeplearning4j/issues/8946
            "non_max_suppression_v4/.*","non_max_suppression_v5/.*",


            // 18.05.2020 - :wq:wq

            "random_uniform_int/.*",
            "random_uniform/.*",
            "random_poisson_v2/.*"
    };

    /* As per TFGraphTestList.printArraysDebugging - this field defines a set of regexes for test cases that should have
       all arrays printed during execution.
       If a test name matches any regex here, an ExecPrintListener will be added to the listeners, and all output
       arrays will be printed during execution
     */
    private final List<String> debugModeRegexes = Arrays.asList("fused_batch_norm/float16_nhwc");



    private ExtensionContext.Store getStore(ExtensionContext context) {
        return context.getStore(ExtensionContext.Namespace.create(getClass(), context.getRequiredTestMethod()));
    }

    @Override
    public ConditionEvaluationResult evaluateExecutionCondition(ExtensionContext extensionContext) {
        String modelName = getStore(extensionContext).toString();
        if(EXECUTE_ONLY_MODELS.isEmpty()) {
            for(String s : IGNORE_REGEXES)  {
                if(modelName.matches(s)) {
                    log.info("\n\tIGNORE MODEL ON REGEX: {} - regex {}", modelName, s);
                    assumeFalse(true);
                }
            }
        } else if(!EXECUTE_ONLY_MODELS.contains(modelName)) {
            log.info("Not executing " + modelName);
            assumeFalse(true);
            //OpValidationSuite.ignoreFailing();
        }


        return ConditionEvaluationResult.disabled("Method found");
    }
}
