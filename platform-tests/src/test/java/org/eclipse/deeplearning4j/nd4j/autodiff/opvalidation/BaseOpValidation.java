/*
 *  ******************************************************************************
 *  *
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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import org.junit.jupiter.api.BeforeEach;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

public abstract class BaseOpValidation extends BaseNd4jTestWithBackends {

    private DataType initialType = Nd4j.dataType();

    // Track whether the GPU context is known to be corrupted
    private static volatile boolean gpuContextCorrupted = false;


    @Override
    public char ordering() {
        return 'c';
    }

    @BeforeEach
    public void beforeClass() {
        // If a previous test corrupted the GPU context, skip subsequent tests
        // This prevents cascade failures that produce misleading error messages
        assumeTrue(!gpuContextCorrupted, "Skipping test - GPU context was corrupted by a previous test failure");

        // Clear any previous error state to prevent test propagation failures
        // This is critical for GPU tests where one test failure can corrupt the CUDA context
        Nd4j.getNativeOps().clearLastError();

        // Try to create a simple array to verify GPU is in a good state
        try {
            INDArray test = Nd4j.create(1);
            // Also verify we can actually use it
            test.assign(1.0);
            // Cleanup
            if (test.closeable()) {
                test.close();
            }
        } catch (Exception e) {
            // If allocation fails after clearing errors, the GPU context is truly corrupted
            gpuContextCorrupted = true;
            assumeTrue(false, "GPU context is corrupted and cannot recover: " + e.getMessage() +
                    ". This is typically caused by a CUDA error in a previous test. " +
                    "Restart the test run to get a clean GPU state.");
        }

        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        Nd4j.getRandom().setSeed(123);
    }

    /**
     * Reset the GPU corrupted flag - useful for test isolation
     */
    public static void resetGpuCorruptedFlag() {
        gpuContextCorrupted = false;
    }
}
