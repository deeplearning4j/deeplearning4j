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

package org.eclipse.deeplearning4j.nd4j.linalg.api;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.HybridDataBuffer;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for HybridDataBuffer CPU device support.
 *
 * These tests verify that HybridDataBuffer implementations properly handle
 * CPU device operations including:
 * 1. HybridDataBuffer interface implementation check
 * 2. Device ownership tracking
 * 3. Basic device pinning operations
 * 4. Data validity management
 *
 * @author Adam Gibson
 */
@Slf4j
@Tag(TagNames.MULTI_THREADED)
@NativeTag
@DisplayName("HybridDataBuffer CPU Device Support Tests")
public class HybridDataBufferCpuDeviceTest extends BaseNd4jTestWithBackends {

    private AffinityManager affinityManager;

    @Override
    public char ordering() {
        return 'c';
    }

    @BeforeEach
    public void setup() {
        affinityManager = Nd4j.getAffinityManager();
    }

    // ========================================================================
    // Section 1: HybridDataBuffer Interface Implementation Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("DataBuffer should implement HybridDataBuffer interface")
    public void testDataBufferImplementsHybridDataBuffer(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        DataBuffer buffer = arr.data();

        assertTrue(buffer instanceof HybridDataBuffer,
                "DataBuffer should implement HybridDataBuffer interface");
    }

    // ========================================================================
    // Section 2: Device Ownership Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("getOwnerDevice should return valid device")
    public void testGetOwnerDevice(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        DataBuffer buffer = arr.data();

        if (buffer instanceof HybridDataBuffer) {
            HybridDataBuffer hybrid = (HybridDataBuffer) buffer;
            DeviceDescriptor owner = hybrid.getOwnerDevice();

            assertNotNull(owner, "Owner device should not be null");
            assertNotNull(owner.getDeviceType(), "Device type should not be null");

            log.info("Owner device: {} (type: {})", owner.getDeviceId(), owner.getDeviceType());
        }
    }

    // ========================================================================
    // Section 3: Device Pinning Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("isPinned should return false initially")
    public void testIsPinnedInitially(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        DataBuffer buffer = arr.data();

        if (buffer instanceof HybridDataBuffer) {
            HybridDataBuffer hybrid = (HybridDataBuffer) buffer;
            assertFalse(hybrid.isPinned(), "Should not be pinned initially");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("getPinnedDevice should return null initially")
    public void testGetPinnedDeviceInitially(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        DataBuffer buffer = arr.data();

        if (buffer instanceof HybridDataBuffer) {
            HybridDataBuffer hybrid = (HybridDataBuffer) buffer;
            DeviceDescriptor pinned = hybrid.getPinnedDevice();
            assertNull(pinned, "Pinned device should be null initially");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("pinTo current device should work")
    public void testPinToCurrentDevice(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        DataBuffer buffer = arr.data();

        if (buffer instanceof HybridDataBuffer) {
            HybridDataBuffer hybrid = (HybridDataBuffer) buffer;

            // Pin to current device (the device where data was created)
            DeviceDescriptor currentDevice = affinityManager.getCurrentDeviceDescriptor();
            assertDoesNotThrow(() -> hybrid.pinTo(currentDevice),
                    "pinTo current device should not throw");

            assertTrue(hybrid.isPinned(), "Should be pinned after pinTo");

            DeviceDescriptor pinned = hybrid.getPinnedDevice();
            assertNotNull(pinned, "Pinned device should not be null after pinning");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("unpin should clear pinned state")
    public void testUnpin(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        DataBuffer buffer = arr.data();

        if (buffer instanceof HybridDataBuffer) {
            HybridDataBuffer hybrid = (HybridDataBuffer) buffer;

            // Pin to current device first
            DeviceDescriptor currentDevice = affinityManager.getCurrentDeviceDescriptor();
            hybrid.pinTo(currentDevice);
            assertTrue(hybrid.isPinned());

            // Unpin
            hybrid.unpin();
            assertFalse(hybrid.isPinned(), "Should not be pinned after unpin");
        }
    }

    // ========================================================================
    // Section 4: Data Validity Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("isValidOn should be true for owner device")
    public void testIsValidOnOwnerDevice(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        DataBuffer buffer = arr.data();

        if (buffer instanceof HybridDataBuffer) {
            HybridDataBuffer hybrid = (HybridDataBuffer) buffer;
            DeviceDescriptor owner = hybrid.getOwnerDevice();

            // Data should be valid on its owner device
            assertTrue(hybrid.isValidOn(owner),
                    "Data should be valid on owner device");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("ensureAvailableOn owner device should not throw")
    public void testEnsureAvailableOnOwner(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        DataBuffer buffer = arr.data();

        if (buffer instanceof HybridDataBuffer) {
            HybridDataBuffer hybrid = (HybridDataBuffer) buffer;
            DeviceDescriptor owner = hybrid.getOwnerDevice();

            assertDoesNotThrow(() -> hybrid.ensureAvailableOn(owner),
                    "ensureAvailableOn owner should not throw");
        }
    }

    // ========================================================================
    // Section 5: Operations After Device Operations
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Array operations should work after device ownership checks")
    public void testOperationsAfterDeviceChecks(Nd4jBackend backend) {
        double[] expectedData = {1, 2, 3, 4, 5};
        INDArray arr = Nd4j.create(expectedData);
        DataBuffer buffer = arr.data();

        if (buffer instanceof HybridDataBuffer) {
            HybridDataBuffer hybrid = (HybridDataBuffer) buffer;

            // Do various device operations
            hybrid.getOwnerDevice();
            hybrid.isPinned();

            // Ensure data is still valid
            hybrid.ensureAvailableOn(hybrid.getOwnerDevice());

            // Array operations should still work
            assertEquals(15.0, arr.sumNumber().doubleValue(), 1e-6);
            assertEquals(1.0, arr.getDouble(0), 1e-6);
            assertEquals(5.0, arr.getDouble(4), 1e-6);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Array operations should work after pinning to current device")
    public void testOperationsAfterPinning(Nd4jBackend backend) {
        INDArray arr = Nd4j.ones(DataType.DOUBLE, 100);
        DataBuffer buffer = arr.data();

        if (buffer instanceof HybridDataBuffer) {
            HybridDataBuffer hybrid = (HybridDataBuffer) buffer;
            hybrid.pinTo(affinityManager.getCurrentDeviceDescriptor());

            // Operations should still work
            INDArray result = arr.add(1.0).mul(2.0);

            assertEquals(400.0, result.sumNumber().doubleValue(), 1e-6,
                    "Operations should produce correct result after pinning");
        }
    }

    // ========================================================================
    // Section 6: Data Type Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("HybridDataBuffer should work with different data types")
    public void testDifferentDataTypes(Nd4jBackend backend) {
        DataType[] types = {DataType.FLOAT, DataType.DOUBLE, DataType.INT, DataType.LONG};

        for (DataType type : types) {
            try {
                INDArray arr = Nd4j.linspace(1, 10, 10, type);
                DataBuffer buffer = arr.data();

                assertTrue(buffer instanceof HybridDataBuffer,
                        "Buffer should be HybridDataBuffer for type " + type);

                if (buffer instanceof HybridDataBuffer) {
                    HybridDataBuffer hybrid = (HybridDataBuffer) buffer;

                    // Owner should be valid
                    DeviceDescriptor owner = hybrid.getOwnerDevice();
                    assertNotNull(owner, "Owner should not be null for type " + type);

                    // Data should be valid on owner
                    assertTrue(hybrid.isValidOn(owner),
                            "Data should be valid on owner for type " + type);
                }
            } catch (Exception e) {
                log.warn("Type {} not fully supported: {}", type, e.getMessage());
            }
        }
    }

    // ========================================================================
    // Section 7: Edge Cases
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Scalar array should work with HybridDataBuffer")
    public void testScalar(Nd4jBackend backend) {
        INDArray scalar = Nd4j.scalar(42.0);
        DataBuffer buffer = scalar.data();

        assertTrue(buffer instanceof HybridDataBuffer,
                "Scalar buffer should be HybridDataBuffer");

        if (buffer instanceof HybridDataBuffer) {
            HybridDataBuffer hybrid = (HybridDataBuffer) buffer;

            DeviceDescriptor owner = hybrid.getOwnerDevice();
            assertNotNull(owner);

            // Should be able to read data
            assertEquals(42.0, scalar.getDouble(0), 1e-6);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Large array should work with HybridDataBuffer")
    public void testLargeArray(Nd4jBackend backend) {
        INDArray large = Nd4j.rand(DataType.DOUBLE, 1000, 1000);
        DataBuffer buffer = large.data();
        double originalSum = large.sumNumber().doubleValue();

        if (buffer instanceof HybridDataBuffer) {
            HybridDataBuffer hybrid = (HybridDataBuffer) buffer;

            // Get owner
            DeviceDescriptor owner = hybrid.getOwnerDevice();
            assertNotNull(owner);

            // Ensure available
            hybrid.ensureAvailableOn(owner);

            // Sum should be unchanged
            assertEquals(originalSum, large.sumNumber().doubleValue(), 1e-3,
                    "Large array data should be preserved");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("View should share buffer properties")
    public void testView(Nd4jBackend backend) {
        INDArray original = Nd4j.linspace(1, 100, 100, DataType.DOUBLE).reshape(10, 10);
        INDArray view = original.getRow(5);
        DataBuffer originalBuffer = original.data();

        if (originalBuffer instanceof HybridDataBuffer) {
            HybridDataBuffer hybrid = (HybridDataBuffer) originalBuffer;
            DeviceDescriptor owner = hybrid.getOwnerDevice();

            // Owner should be valid
            assertNotNull(owner, "Owner should not be null");

            // View should be able to access data - just verify no exceptions thrown
            assertDoesNotThrow(() -> view.sumNumber(),
                    "View should be able to compute sum without throwing");
        }
    }
}
