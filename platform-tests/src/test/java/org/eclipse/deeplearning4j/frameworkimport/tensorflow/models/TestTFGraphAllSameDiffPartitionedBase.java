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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.provider.Arguments;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Stream;
@Slf4j
@Tag(TagNames.TENSORFLOW)
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
            // Missing test data files - inputs not loaded
            "emptyArrayTests/unstack/.*",
            "emptyArrayTests/zeros_like/.*",
            "emptyArrayTests/ones_like/.*",
            "emptyArrayTests/identity_n/.*",
            // reverse/ReverseV2 missing test data files
            "reverse/.*",
            // space_to_batch padding issues
            "space_to_batch/rank4nhwc_pad.*",
            // RNN/control flow variable resolution issues
            "lstm_mnist.*",
            "primitive_gru.*",
            "primitive_lstm.*",
            "partition_stitch_misc.*",
            "while2/.*",
            // SVD and tensor_array implementation issues
            "svd/.*",
            "tensor_array/.*",
            // topk and atan2 implementation issues
            "topk/.*",
            "transforms/atan2.*",
            // logicaland/logicalor shape mismatch
            "transforms/logicaland.*",
            "transforms/logicalor.*",
            "transforms/logicalnot_.*",
            "transforms/logicalxor.*",
            // unsorted_segment DataBuffer issues
            "unsorted_segment/.*",
            // yiq_to_rgb implementation issues
            "yiq_to_rgb/.*",
            // yuv_to_rgb tensordot issues
            "yuv_to_rgb/.*",
            // sparse_softmax_ce DataBuffer issues
            "losses/sparse_softmax_ce.*",
            // triangular_solve excluded from CPU backend
            "triangular_solve/emptyArrayTest.*",
            // Missing test data files for dilated convolutions
            "cnn1d_layers/.*_d2_.*",
            "cnn2d_layers/.*_d2_.*",
            "cnn2d_layers/.*_d12_.*",
            // conv_1 and conv_2 have corrupted test data files (wrong input shapes)
            "conv_1.*",
            "conv_2.*",
            // Missing test data files for cnn3d, conv2d_transpose, cond tests
            "cnn3d_layers/.*",
            "conv2d_transpose/.*",
            "cond/.*",
            // TF control flow variable resolution issues
            "embedding_lookup/.*",
            // in_top_k implementation issues
            "in_top_k/.*",
            // DataBuffer integrity issues
            "g_02.*",
            "g_09.*",
            "g_10.*",
            // g_07 has missing test data files
            "g_07.*",
            // lu, matmul rank5, matrix_* implementation issues
            "lu/.*",
            "matmul/rank5.*",
            "matrix_band_part/.*",
            "matrix_diag_part/.*",
            "matrix_inverse/.*",
            // nth_element implementation issues
            "nth_element/.*",
            // DynamicPartition, RNN variable resolution issues
            "rnn/.*",
            // reverse/ReverseV2 implementation issues
            "reverse/rank1.*",
            // bitcast implementation issues
            "bitcast/.*",
            // segment_* implementation issues
            "segment/.*",
            // sepconv1d implementation issues
            "sepconv1d_layers/.*",
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