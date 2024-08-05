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
import java.util.Map;
import java.util.stream.Stream;


public class TestTFGraphAllSameDiffPartitioned36 extends TestTFGraphAllSameDiffPartitionedBase {



    @ParameterizedTest
    @MethodSource("generateTests")
    public void runTest(Map<String, INDArray> inputs, Map<String, INDArray> predictions, String modelName, File localTestDir) throws Exception {
        super.runTest(inputs, predictions, modelName, localTestDir, 36);
    }

    public static Stream<Arguments> generateTests() throws IOException {
        return generateTestsForPartition(36);
    }
}