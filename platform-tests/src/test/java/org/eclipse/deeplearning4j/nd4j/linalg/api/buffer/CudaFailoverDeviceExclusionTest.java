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

package org.eclipse.deeplearning4j.nd4j.linalg.api.buffer;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.NativeOpsHolder;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for the CudaMemoryPool device exclusion API.
 *
 * The exclusion API allows subprocesses to declare which GPU devices should
 * never receive failover allocations. This prevents cross-process memory
 * interference: e.g., the staging process excludes the embedding device so
 * its failover allocations don't displace the embedding subprocess's
 * resident pages via cudaMallocManaged UVA.
 */
@Slf4j
@DisplayName("CUDA Memory Pool Device Exclusion Tests")
public class CudaFailoverDeviceExclusionTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    private boolean isCudaBackend() {
        return Nd4j.getBackend().getClass().getSimpleName().toLowerCase().contains("cuda");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Exclusion API is idempotent and clearable")
    public void testExclusionApiIdempotent(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Requires CUDA backend");
        var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        // Double-add should not throw
        nativeOps.addExcludedFailoverDevice(1);
        nativeOps.addExcludedFailoverDevice(1);

        // Should report excluded
        assertTrue(nativeOps.isDeviceExcludedFromFailover(1));

        // Remove should work
        nativeOps.removeExcludedFailoverDevice(1);
        assertFalse(nativeOps.isDeviceExcludedFromFailover(1));

        // Clear should work even when empty
        nativeOps.clearExcludedFailoverDevices();
        assertFalse(nativeOps.isDeviceExcludedFromFailover(0));
        assertFalse(nativeOps.isDeviceExcludedFromFailover(1));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Multiple devices can be excluded independently")
    public void testMultipleDeviceExclusion(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Requires CUDA backend");
        var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        try {
            nativeOps.addExcludedFailoverDevice(0);
            nativeOps.addExcludedFailoverDevice(2);

            assertTrue(nativeOps.isDeviceExcludedFromFailover(0));
            assertFalse(nativeOps.isDeviceExcludedFromFailover(1));
            assertTrue(nativeOps.isDeviceExcludedFromFailover(2));

            // Remove one, other stays
            nativeOps.removeExcludedFailoverDevice(0);
            assertFalse(nativeOps.isDeviceExcludedFromFailover(0));
            assertTrue(nativeOps.isDeviceExcludedFromFailover(2));
        } finally {
            nativeOps.clearExcludedFailoverDevices();
        }
    }

}
