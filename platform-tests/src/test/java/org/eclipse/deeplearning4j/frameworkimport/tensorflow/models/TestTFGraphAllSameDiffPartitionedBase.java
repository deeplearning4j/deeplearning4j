/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.frameworkimport.tensorflow.models;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.frameworkimport.tensorflow.TFGraphTestAllHelper;
import org.eclipse.deeplearning4j.tests.extensions.FailFast;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.provider.Arguments;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Stream;
@Slf4j
@Tag(TagNames.TENSORFLOW)
@ExtendWith(FailFast.class)
public abstract class TestTFGraphAllSameDiffPartitionedBase {

    public static final TFGraphTestAllHelper.ExecuteWith EXECUTE_WITH = TFGraphTestAllHelper.ExecuteWith.SAMEDIFF;
    public static final String BASE_DIR = "tf_graphs/examples";
    public static final String MODEL_FILENAME = "frozen_model.pb";
    public static final int TOTAL_TESTS = 1918;
    public static final int TESTS_PER_PARTITION = 50;

    public final static List<String> EXECUTE_ONLY_MODELS = Arrays.asList(
    );

    public static final String[] IGNORE_REGEXES = new String[] {
            //inputs don't even run with tf-java
            "simplewhile_0",
            "simplewhile_1",
            "simplewhile_0_alt",
            "simpleif_0",
            "simple_while",
            "simpleif_0_alt",
            "simplewhile_nested",
            "simple_cond",
            //doesn't execute in tf java or nd4j, ignoring
           "ragged/identity/2d",
            "ragged/add/2d",
           //same as below: when running in tf java, the results are actually equal. The python execution saved results look to be wrong.
            "norm_tests/norm_7",
            //when running in tf java, the results are actually equal. The python execution saved results look to be wrong.
            "non2d_0",
            //invalid graph: tries to multiply 2 invalid shapes
            "non2d_1",
            "non2d_0A",
            //tf-java contradicts the results that we load from python. Ignoring.
            "fused_batch_norm/float32_nhwc",
            "fused_batch_norm/float32_nhcw",
            "non_max_suppression_v4/float16_with_thresholds",
            "non_max_suppression_v4/float32_with_thresholds",
            "non_max_suppression_v4/float32_with_thresholds_pad_to_max_output_size",
            "non_max_suppression_v5/.*",
            "resize_bicubic/float64",
            "resize_bicubic/int32",
            "multinomial/.*",
            "reductions/scatter_update_vector",
            "reductions/scatter_update_scalar",
            "emptyArrayTests/scatter_update/rank1_emptyIndices_emptyUpdates",
            "bincount/rank2_weights",
            "slogdet/.*",
            "fused_batch_norm/float16_nhwc",
            "emptyArrayTests/scatter_update/rank2_emptyIndices_emptyUpdates",
            "layers_dropout/.*",
            "truncatemod/.*",
            "confusion/.*",
            "conv_4",
            "conv3d_transpose.*",
            "ragged/reduce_mean/.*",
            "random_gamma/.*",
            "Conv3DBackpropInputV2/.*",
            "random_uniform_int/.*",
            "random_uniform/.*",
            "random_poisson_v2/.*",
            "random_poisson/.*",
    };

    private static final List<String> debugModeRegexes = Arrays.asList(
            // Specify debug mode regexes, if any
    );




    public  void runTest(Map<String, INDArray> inputs, Map<String, INDArray> predictions, String modelName, File localTestDir, int partitionIndex) throws Exception {
        TestRunner testRunner = new TestRunner(debugModeRegexes);
        testRunner.runTest(inputs, predictions, modelName, localTestDir);
    }

    public static Stream<Arguments> generateTestsForPartition(int partitionIndex) throws IOException {
        int startIdx = partitionIndex * TESTS_PER_PARTITION;
        int endIdx = Math.min(startIdx + TESTS_PER_PARTITION, TOTAL_TESTS);
        if(!EXECUTE_ONLY_MODELS.isEmpty()) {
            startIdx = 0;
            endIdx = EXECUTE_ONLY_MODELS.size();
        }
        List<Object[]> params = fetchData(startIdx, endIdx);
        List<Object[]> partitionedParams = params;

        List<Arguments> argumentsList = new ArrayList<>();
        for (Object[] partitionedParam : partitionedParams) {
            argumentsList.add(Arguments.of(partitionedParam));
        }

        return argumentsList.stream();
    }

    public static List<Object[]> fetchData(int startIdx, int endIdx) throws IOException {
        String localPath = System.getenv(TFGraphTestAllHelper.resourceFolderVar);
        File baseDir;
        if (localPath == null) {
            baseDir = new File(System.getProperty("java.io.tmpdir"), UUID.randomUUID().toString());
        } else {
            baseDir = new File(localPath);
        }
        return TFGraphTestAllHelper.fetchTestParams(BASE_DIR, MODEL_FILENAME, EXECUTE_WITH, baseDir, startIdx, endIdx);
    }



}