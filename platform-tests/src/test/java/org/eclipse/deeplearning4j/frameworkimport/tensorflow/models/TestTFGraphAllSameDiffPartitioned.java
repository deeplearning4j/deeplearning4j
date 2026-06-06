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

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Consolidated parameterized test class replacing 38 individual partition shim classes
 * (TestTFGraphAllSameDiffPartitioned0 through TestTFGraphAllSameDiffPartitioned38,
 * excluding partition 1 which was never present).
 *
 * Each test argument includes the partition index so the base class can route to the
 * correct slice of the test corpus.
 */
public class TestTFGraphAllSameDiffPartitioned extends TestTFGraphAllSameDiffPartitionedBase {

    // All valid partition indices: 0, 2-38 (partition 1 was never present)
    private static final int[] PARTITION_INDICES = buildPartitionIndices();

    private static int[] buildPartitionIndices() {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i <= 38; i++) {
            if (i == 1) continue; // partition 1 was never present
            indices.add(i);
        }
        return indices.stream().mapToInt(Integer::intValue).toArray();
    }

    @ParameterizedTest
    @MethodSource("generateTests")
    public void runTest(Map<String, INDArray> inputs, Map<String, INDArray> predictions,
                        String modelName, File localTestDir, int partitionIndex) throws Exception {
        super.runTest(inputs, predictions, modelName, localTestDir, partitionIndex);
    }

    public static Stream<Arguments> generateTests() throws IOException {
        List<Arguments> allArgs = new ArrayList<>();
        for (int partitionIndex : PARTITION_INDICES) {
            Stream<Arguments> partitionArgs = generateTestsForPartition(partitionIndex);
            partitionArgs.forEach(args -> {
                // Each args from generateTestsForPartition has: (inputs, predictions, modelName, localTestDir)
                // Append the partitionIndex so runTest can dispatch correctly.
                Object[] original = args.get();
                Object[] augmented = new Object[original.length + 1];
                System.arraycopy(original, 0, augmented, 0, original.length);
                augmented[original.length] = partitionIndex;
                allArgs.add(Arguments.of(augmented));
            });
        }
        return allArgs.stream();
    }
}
