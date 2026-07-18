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

package org.eclipse.deeplearning4j.nd4j.backends;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.factory.DataBufferFactory;
import org.nd4j.linalg.factory.Nd4j;

import java.io.InputStream;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * VulkanDataBufferFactoryTest — factory-loads contract for ADR-0112 §1 Step 5/6.
 *
 * <p>Tests the properties flip: {@code databufferfactory} and
 * {@code ndarrayfactory.class} now resolve to Vulkan-namespaced classes.
 * The factory-contract leg runs on any box (0-device CI included); device
 * legs are assume-gated behind a real Vulkan device.</p>
 *
 * <p>Design verification: confirms that the thin-subclass (Option A) design
 * works — VulkanDataBufferFactory instantiates, produces valid DataBuffers,
 * and the smoke test (VulkanBackendSmokeTest) stays green with the flipped keys.</p>
 *
 * <p>Run with:
 * <pre>
 *   cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test -Ptest-vulkan \
 *     -Dtest=VulkanDataBufferFactoryTest 2>&1 | tee /tmp/vulkan-factory-test.log
 * </pre>
 */
@Slf4j
@Tag(TagNames.VULKAN)
@DisplayName("VulkanDataBufferFactory — properties flip + factory contract (ADR-0112 §1)")
public class VulkanDataBufferFactoryTest {

    private static final String VULKAN_BACKEND_CLASS   = "org.nd4j.linalg.vulkan.VulkanBackend";
    private static final String VULKAN_BINDINGS_CLASS  = "org.nd4j.linalg.vulkan.bindings.Nd4jVulkan";
    private static final String VULKAN_FACTORY_CLASS   = "org.nd4j.linalg.vulkan.VulkanDataBufferFactory";
    private static final String VULKAN_NDARRAY_FACTORY = "org.nd4j.linalg.vulkan.VulkanNDArrayFactory";
    private static final String LINALG_PROPS           = "/nd4j-vulkan.properties";

    private static boolean vulkanClassAvailable = false;
    private static boolean vulkanDevicePresent  = false;

    @BeforeAll
    static void probeVulkan() {
        try {
            Class.forName(VULKAN_BACKEND_CLASS);
            vulkanClassAvailable = true;
        } catch (ClassNotFoundException e) {
            log.warn("VulkanDataBufferFactoryTest: VulkanBackend not on classpath — run with -Ptest-vulkan");
            return;
        }
        try {
            Class<?> bindings = Class.forName(VULKAN_BINDINGS_CLASS);
            Object nativeOps  = bindings.getDeclaredConstructor().newInstance();
            int count = (int) bindings.getMethod("getAvailableDevices").invoke(nativeOps);
            vulkanDevicePresent = (count > 0);
            log.info("VulkanDataBufferFactoryTest: vulkanClassAvailable=true, deviceCount={}", count);
        } catch (Exception e) {
            log.info("VulkanDataBufferFactoryTest: bindings unavailable ({}) — device legs skip",
                    e.getClass().getSimpleName());
            vulkanDevicePresent = false;
        }
    }

    private static void requireVulkanClass() {
        assumeTrue(vulkanClassAvailable,
                "VulkanBackend not on classpath — run with -Ptest-vulkan.");
    }

    private static void requireVulkanDevice() {
        requireVulkanClass();
        assumeTrue(vulkanDevicePresent,
                "No Vulkan device enumerated — device legs require a Vulkan ICD (lavapipe or real GPU).");
    }

    // =========================================================================
    // CONTRACT LEGS: run on any box (including 0-device CI)
    // =========================================================================

    /**
     * The properties file must have BOTH factory keys pointing at Vulkan-namespaced
     * classes — no residual cpu.nativecpu.* references for databufferfactory or
     * ndarrayfactory.class.
     *
     * <p>This is the "properties diff" verification required by the task spec:
     * confirms the flip is complete and neither key regresses to cpu.nativecpu.*.</p>
     */
    @Test
    @DisplayName("(contract) nd4j-vulkan.properties: factory keys point at Vulkan classes")
    void testPropertiesKeysFlipped() throws Exception {
        requireVulkanClass();

        Class<?> backendClass = Class.forName(VULKAN_BACKEND_CLASS);
        Properties props = new Properties();
        try (InputStream is = backendClass.getResourceAsStream(LINALG_PROPS)) {
            assertNotNull(is, LINALG_PROPS + " must be packaged in the nd4j-vulkan jar");
            props.load(is);
        }

        // databufferfactory must not be the cpu.nativecpu stub any more.
        String dbf = props.getProperty("databufferfactory");
        assertNotNull(dbf, "databufferfactory key must be present in nd4j-vulkan.properties");
        assertEquals(VULKAN_FACTORY_CLASS, dbf,
                "databufferfactory must point at VulkanDataBufferFactory, not the cpu.nativecpu stub. "
                        + "Got: " + dbf);
        assertFalse(dbf.contains("cpu.nativecpu"),
                "databufferfactory must not contain cpu.nativecpu: " + dbf);
        log.info("databufferfactory = {}", dbf);

        // ndarrayfactory.class must not be the cpu.nativecpu stub any more.
        String naf = props.getProperty("ndarrayfactory.class");
        assertNotNull(naf, "ndarrayfactory.class key must be present in nd4j-vulkan.properties");
        assertEquals(VULKAN_NDARRAY_FACTORY, naf,
                "ndarrayfactory.class must point at VulkanNDArrayFactory, not the cpu.nativecpu stub. "
                        + "Got: " + naf);
        assertFalse(naf.contains("cpu.nativecpu"),
                "ndarrayfactory.class must not contain cpu.nativecpu: " + naf);
        log.info("ndarrayfactory.class = {}", naf);

        // real.class.double stays BaseNDArray — no Vulkan NDArray subclass yet (ADR-0112 §1).
        String rcd = props.getProperty("real.class.double");
        assertNotNull(rcd, "real.class.double key must be present in nd4j-vulkan.properties");
        log.info("real.class.double = {} (BaseNDArray is correct per ADR-0112 §1)", rcd);
    }

    /**
     * VulkanDataBufferFactory must be loadable from the classpath (i.e. the class
     * exists, is accessible, and instantiates without error) regardless of whether
     * a Vulkan device is present.  This is the "factory class instantiates" contract
     * required by the task spec.
     */
    @Test
    @DisplayName("(contract) VulkanDataBufferFactory is loadable and instantiable")
    void testFactoryClassInstantiates() throws Exception {
        requireVulkanClass();

        Class<?> factoryClass = assertDoesNotThrow(
                () -> Class.forName(VULKAN_FACTORY_CLASS),
                "VulkanDataBufferFactory must be on the classpath when -Ptest-vulkan is active");
        assertNotNull(factoryClass, "Class.forName returned null for " + VULKAN_FACTORY_CLASS);

        // Must instantiate without error — the Nd4j init path does this reflectively.
        Object factory = assertDoesNotThrow(
                () -> factoryClass.getDeclaredConstructor().newInstance(),
                "VulkanDataBufferFactory() no-arg constructor must not throw");
        assertNotNull(factory, "No-arg constructor returned null");
        assertTrue(factory instanceof DataBufferFactory,
                "VulkanDataBufferFactory must implement DataBufferFactory. "
                        + "Actual supertype chain: " + factoryClass.getSuperclass().getName());
        log.info("VulkanDataBufferFactory instantiated as: {}", factory.getClass().getName());
    }

    /**
     * VulkanNDArrayFactory must also be loadable and instantiable (companion to the
     * databufferfactory flip).
     */
    @Test
    @DisplayName("(contract) VulkanNDArrayFactory is loadable and instantiable")
    void testNDArrayFactoryClassInstantiates() throws Exception {
        requireVulkanClass();

        Class<?> nfClass = assertDoesNotThrow(
                () -> Class.forName(VULKAN_NDARRAY_FACTORY),
                "VulkanNDArrayFactory must be on the classpath when -Ptest-vulkan is active");
        assertNotNull(nfClass, "Class.forName returned null for " + VULKAN_NDARRAY_FACTORY);

        Object nf = assertDoesNotThrow(
                () -> nfClass.getDeclaredConstructor().newInstance(),
                "VulkanNDArrayFactory() no-arg constructor must not throw");
        assertNotNull(nf, "No-arg constructor returned null");
        log.info("VulkanNDArrayFactory instantiated as: {}", nf.getClass().getName());
    }

    /**
     * VulkanDataBufferFactory.createFloat(length) must produce a valid non-null DataBuffer.
     * This verifies that the thin-subclass delegation to DefaultDataBufferFactory works —
     * a factory that crashes on the first buffer creation is broken regardless of design
     * rationale.
     */
    @Test
    @DisplayName("(contract) VulkanDataBufferFactory.createFloat produces a valid DataBuffer")
    void testFactoryCreateFloatBuffer() throws Exception {
        requireVulkanClass();

        Class<?> factoryClass = Class.forName(VULKAN_FACTORY_CLASS);
        DataBufferFactory factory = (DataBufferFactory) factoryClass.getDeclaredConstructor().newInstance();

        DataBuffer buf = assertDoesNotThrow(
                () -> factory.createFloat(16L),
                "VulkanDataBufferFactory.createFloat(16) must not throw");
        assertNotNull(buf, "createFloat(16) must not return null");
        assertEquals(16, buf.length(), "Buffer length must match requested length");
        assertEquals(DataType.FLOAT, buf.dataType(), "Buffer dtype must be FLOAT");
        log.info("VulkanDataBufferFactory.createFloat(16): length={} dtype={}", buf.length(), buf.dataType());
    }

    /**
     * VulkanDataBufferFactory.create(DataType, length, initialize) must handle all
     * common dtypes without throwing.  This catches typos in the thin-subclass
     * constructor or any accidental method override that breaks delegation.
     */
    @Test
    @DisplayName("(contract) VulkanDataBufferFactory.create handles common dtypes")
    void testFactoryCreateCommonDtypes() throws Exception {
        requireVulkanClass();

        Class<?> factoryClass = Class.forName(VULKAN_FACTORY_CLASS);
        DataBufferFactory factory = (DataBufferFactory) factoryClass.getDeclaredConstructor().newInstance();

        for (DataType dt : new DataType[]{DataType.FLOAT, DataType.DOUBLE, DataType.INT,
                DataType.LONG, DataType.HALF, DataType.BOOL}) {
            DataBuffer buf = assertDoesNotThrow(
                    () -> factory.create(dt, 8L, false),
                    "create(" + dt + ", 8, false) must not throw");
            assertNotNull(buf, "create(" + dt + ", 8, false) must not return null");
            assertEquals(dt, buf.dataType(),
                    "Buffer dtype must match requested dtype for " + dt);
            log.info("  create({}, 8, false): OK — length={}", dt, buf.length());
        }
    }

    /**
     * Backend self-selection smoke: VulkanBackend.canRun() must still evaluate
     * correctly with the flipped factory keys.  On a 0-device box canRun()==false;
     * on a device box canRun()==true.  The factory class flip must not break the
     * canRun() gate (e.g. by causing a ClassNotFoundException during backend init).
     */
    @Test
    @DisplayName("(contract) VulkanBackend.canRun() is unchanged after factory key flip")
    void testBackendCanRunUnchangedAfterFlip() throws Exception {
        requireVulkanClass();

        Class<?> backendClass = Class.forName(VULKAN_BACKEND_CLASS);
        Object backend = backendClass.getDeclaredConstructor().newInstance();
        boolean canRun = (Boolean) backendClass.getMethod("canRun").invoke(backend);
        log.info("VulkanBackend.canRun() after factory key flip = {}", canRun);

        // Verify canRun() == (device present) — the contract from VulkanBackendSmokeTest.
        // If the factory flip had broken backend init, canRun() would throw, not return false.
        boolean expected = vulkanDevicePresent;
        assertEquals(expected, canRun,
                "VulkanBackend.canRun() must equal (deviceCount >= 1) after factory key flip. "
                        + "If this fails with an exception rather than a value mismatch, "
                        + "the factory class flip broke backend initialisation.");
    }

    // =========================================================================
    // DEVICE LEGS: require Vulkan device
    // =========================================================================

    /**
     * When a Vulkan device is present, Nd4j.createBuffer(DataType.FLOAT, 32, false)
     * must produce a DataBuffer whose factory class is VulkanDataBufferFactory
     * (i.e. the properties wire-up is active end-to-end).
     *
     * <p>This is the strongest integration check: it exercises the properties →
     * factory instantiation → buffer creation chain that Nd4j.initWithConf() drives.</p>
     */
    @Test
    @DisplayName("(device) Nd4j.getDataBufferFactory() is VulkanDataBufferFactory on a device box")
    void testNd4jUsesVulkanFactory() throws Exception {
        requireVulkanDevice();

        DataBufferFactory factory = Nd4j.getDataBufferFactory();
        assertNotNull(factory, "Nd4j.getDataBufferFactory() must not be null");
        String factoryClass = factory.getClass().getName();
        log.info("Nd4j.getDataBufferFactory() class = {}", factoryClass);

        // Must be our Vulkan factory (or a subclass of it — future-proof).
        assertTrue(factoryClass.startsWith("org.nd4j.linalg.vulkan"),
                "On a Vulkan-device box Nd4j must use a vulkan-namespaced factory. "
                        + "Got: " + factoryClass
                        + " — check that nd4j-vulkan.properties databufferfactory key is active "
                        + "and the Vulkan backend was selected (canRun()==true).");
    }

    /**
     * On a device box, a DataBuffer created via the Vulkan factory path must still
     * support the dual-buffer surface (syncToSpecial, syncToPrimary) without throwing.
     * This is the integration verification of the "thin-subclass is correct because
     * OpaqueDataBuffer dispatches to the native TU" design rationale.
     */
    @Test
    @DisplayName("(device) DataBuffer created via Nd4j (VulkanFactory path) supports dual-buffer sync")
    void testNd4jBufferSupportsDualBufferSync() throws Exception {
        requireVulkanDevice();

        // Allocate via Nd4j (goes through VulkanDataBufferFactory → DefaultDataBufferFactory
        // → BaseDataBuffer → OpaqueDataBuffer → native Vulkan TU).
        DataBuffer buf = Nd4j.createBuffer(DataType.FLOAT, 64L, false);
        assertNotNull(buf, "Nd4j.createBuffer must not return null on a device box");
        assertEquals(DataType.FLOAT, buf.dataType());
        assertEquals(64L, buf.length());

        // The underlying OpaqueDataBuffer must support the dual-buffer surface.
        org.nd4j.nativeblas.OpaqueDataBuffer odb = buf.opaqueBuffer();
        assertNotNull(odb, "DataBuffer.opaqueBuffer() must not be null");
        assertFalse(odb.isNull(), "OpaqueDataBuffer handle must not be null");

        // syncToSpecial / syncToPrimary must not throw.
        assertDoesNotThrow(odb::syncToSpecial,
                "syncToSpecial() must not throw on a Vulkan device box");
        assertDoesNotThrow(odb::syncToPrimary,
                "syncToPrimary() must not throw on a Vulkan device box");

        log.info("(device) DataBuffer via Nd4j: length={} dtype={} — dual-buffer sync OK",
                buf.length(), buf.dataType());
    }
}
