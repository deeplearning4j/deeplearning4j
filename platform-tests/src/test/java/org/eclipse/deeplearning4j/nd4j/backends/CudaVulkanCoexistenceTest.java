/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.backends;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.LongPointer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.MultiBackendNativeOpsHolder;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.OpaqueContext;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.nativeblas.OpaqueNDArray;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.ServiceLoader;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag(TagNames.VULKAN)
@Tag(TagNames.BACKEND_DISCOVERY)
@DisplayName("CUDA and Vulkan same-JVM coexistence")
public class CudaVulkanCoexistenceTest {

    private static final String CUDA_BACKEND = "org.nd4j.linalg.jcublas.JCublasBackend";
    private static final String VULKAN_BACKEND = "org.nd4j.linalg.vulkan.VulkanBackend";

    @Test
    @DisplayName("both hardware backends load and execute kernels in one JVM")
    public void cudaAndVulkanAreRunnableInSameJvm() {
        String requestedOrder = System.getProperty("nd4j.coexistence.loadOrder", "cuda-first");
        assertTrue("cuda-first".equals(requestedOrder) || "vulkan-first".equals(requestedOrder),
                "nd4j.coexistence.loadOrder must be cuda-first or vulkan-first");

        List<String> loadOrder = "vulkan-first".equals(requestedOrder)
                ? List.of(VULKAN_BACKEND, CUDA_BACKEND)
                : List.of(CUDA_BACKEND, VULKAN_BACKEND);
        List<DeviceDescriptor> cudaDevices = null;
        for (String className : loadOrder) {
            Nd4jBackend backend = instantiateBackend(className);
            assertTrue(backend.canRun(), className + " is present but cannot run on this hardware lane");
            List<DeviceDescriptor> devices = backend.discoverDevices();
            assertTrue(!devices.isEmpty(), className + " did not enumerate any native devices");
            if (CUDA_BACKEND.equals(className)) {
                cudaDevices = devices;
            }
        }

        if (Boolean.getBoolean("nd4j.coexistence.requireMultiGpu")) {
            assertTrue(cudaDevices != null && cudaDevices.size() >= 2,
                    "The coexistence lane requires at least two real CUDA devices, but discovered "
                            + (cudaDevices == null ? 0 : cudaDevices.size()));
        }
        Set<String> discovered = new HashSet<>();
        ServiceLoader.load(Nd4jBackend.class).forEach(backend -> {
            String className = backend.getClass().getName();
            if (CUDA_BACKEND.equals(className) || VULKAN_BACKEND.equals(className)) {
                discovered.add(className);
            }
        });
        assertTrue(discovered.contains(CUDA_BACKEND), "CUDA backend was not discovered in the test JVM");
        assertTrue(discovered.contains(VULKAN_BACKEND), "Vulkan backend was not discovered in the test JVM");

        assertTrue(MultiBackendNativeOpsHolder.enableMultiBackend(),
                "NativeOps multi-backend initialization failed");
        MultiBackendNativeOpsHolder holder = MultiBackendNativeOpsHolder.getInstance();
        assertTrue(holder.isBackendAvailable(DeviceType.CUDA_GPU),
                "CUDA NativeOps was not registered");
        assertTrue(holder.isBackendAvailable(DeviceType.VULKAN_GPU),
                "Vulkan NativeOps was not registered");
        assertTrue(holder.getDeviceCount(DeviceType.CUDA_GPU) > 0,
                "CUDA NativeOps reported no devices");
        assertTrue(holder.getDeviceCount(DeviceType.VULKAN_GPU) > 0,
                "Vulkan NativeOps reported no devices");
        if (Boolean.getBoolean("nd4j.coexistence.requireMultiGpu")) {
            assertTrue(holder.getDeviceCount(DeviceType.CUDA_GPU) >= 2,
                    "Multi-backend NativeOps did not retain both CUDA devices");
        }

        NativeOps cudaOps = holder.getOpsForDeviceType(DeviceType.CUDA_GPU);
        NativeOps vulkanOps = holder.getOpsForDeviceType(DeviceType.VULKAN_GPU);
        if (Boolean.getBoolean("nd4j.coexistence.requireTriton")) {
            assertTritonAvailable(cudaOps, "CUDA");
            assertTritonAvailable(vulkanOps, "Vulkan");
        }

        List<String> vulkanNames =
                vulkanDeviceNames(vulkanOps, holder.getDeviceCount(DeviceType.VULKAN_GPU));
        int hardwareVulkanDevice = firstHardwareVulkanDevice(vulkanNames);
        if (Boolean.getBoolean("nd4j.coexistence.requireHardwareVulkan")) {
            assertTrue(hardwareVulkanDevice >= 0,
                    "The coexistence lane requires a hardware Vulkan device; enumerated "
                            + vulkanNames);
        }
        int vulkanDevice = hardwareVulkanDevice >= 0 ? hardwareVulkanDevice : 0;

        for (String className : loadOrder) {
            if (CUDA_BACKEND.equals(className)) {
                executeAddKernel(cudaOps, 0, "CUDA");
            } else {
                executeAddKernel(vulkanOps, vulkanDevice, "Vulkan");
            }
        }
    }

    private static void assertTritonAvailable(NativeOps nativeOps, String backendName) {
        assertTrue(nativeOps.isTritonAvailable(),
                backendName + " was built without the shared Triton DSP/compiler stack "
                        + "on a coexistence lane that requires it");
    }

    private static List<String> vulkanDeviceNames(NativeOps nativeOps, int deviceCount) {
        List<String> names = new ArrayList<>(deviceCount);
        for (int i = 0; i < deviceCount; i++) {
            names.add(nativeOps.getDeviceName(i));
        }
        return names;
    }

    private static int firstHardwareVulkanDevice(List<String> names) {
        for (int i = 0; i < names.size(); i++) {
            if (isHardwareVulkanDevice(names.get(i))) {
                return i;
            }
        }
        return -1;
    }

    private static void executeAddKernel(
            NativeOps nativeOps, int deviceId, String backendName) {
        final int length = 8;
        OpaqueDataBuffer shapeInfo = null;
        OpaqueDataBuffer x = null;
        OpaqueDataBuffer y = null;
        OpaqueDataBuffer z = null;
        OpaqueNDArray xArray = null;
        OpaqueNDArray yArray = null;
        OpaqueNDArray zArray = null;
        OpaqueContext context = null;

        try {
            nativeOps.setDevice(deviceId);
            assertEquals(deviceId, nativeOps.getDevice(),
                    backendName + " did not bind the requested device");

            long[] shape = new long[]{1, length, 1, 0, 1, 'c'};
            ArrayOptionsHelper.setDataTypeInShapeInfo(shape, DataType.FLOAT);
            shapeInfo = nativeOps.allocateDataBuffer(shape.length, DataType.INT64.toInt(), true);
            assertLive(shapeInfo, backendName + " shape buffer");
            new LongPointer(shapeInfo.primaryBuffer()).put(shape);
            nativeOps.dbTickHostWrite(shapeInfo);
            nativeOps.dbSyncToSpecial(shapeInfo);

            x = nativeOps.allocateDataBuffer(length, DataType.FLOAT.toInt(), true);
            y = nativeOps.allocateDataBuffer(length, DataType.FLOAT.toInt(), true);
            z = nativeOps.allocateDataBuffer(length, DataType.FLOAT.toInt(), true);
            assertLive(x, backendName + " x buffer");
            assertLive(y, backendName + " y buffer");
            assertLive(z, backendName + " z buffer");

            FloatPointer xHost = new FloatPointer(x.primaryBuffer());
            FloatPointer yHost = new FloatPointer(y.primaryBuffer());
            for (int i = 0; i < length; i++) {
                xHost.put(i, i + 0.25f);
                yHost.put(i, 2.0f * i + 1.0f);
            }
            nativeOps.dbTickHostWrite(x);
            nativeOps.dbTickHostWrite(y);
            nativeOps.dbSyncToSpecial(x);
            nativeOps.dbSyncToSpecial(y);

            xArray = nativeOps.create(shapeInfo, x, x, 0);
            yArray = nativeOps.create(shapeInfo, y, y, 0);
            zArray = nativeOps.create(shapeInfo, z, z, 0);
            assertLive(xArray, backendName + " x array");
            assertLive(yArray, backendName + " y array");
            assertLive(zArray, backendName + " z array");

            context = nativeOps.createGraphContext(0);
            assertLive(context, backendName + " op context");
            nativeOps.setGraphContextInputArray(context, 0, xArray);
            nativeOps.setGraphContextInputArray(context, 1, yArray);
            nativeOps.setGraphContextOutputArray(context, 0, zArray);

            int status = nativeOps.execCustomOp2(null, new AddOp().opHash(), context);
            assertEquals(0, status, backendName + " add kernel returned status " + status);
            assertEquals(0, nativeOps.lastErrorCode(),
                    backendName + " native error: " + nativeOps.lastErrorMessage());

            nativeOps.dbSyncToPrimary(z);
            FloatPointer zHost = new FloatPointer(z.primaryBuffer());
            for (int i = 0; i < length; i++) {
                assertEquals(3.0f * i + 1.25f, zHost.get(i), 1e-6f,
                        backendName + " add result mismatch at index " + i);
            }
        } finally {
            if (context != null && !context.isNull()) {
                nativeOps.deleteGraphContext(context);
            }
            if (xArray != null && !xArray.isNull()) {
                nativeOps.deleteNDArray(xArray);
            }
            if (yArray != null && !yArray.isNull()) {
                nativeOps.deleteNDArray(yArray);
            }
            if (zArray != null && !zArray.isNull()) {
                nativeOps.deleteNDArray(zArray);
            }
            if (x != null && !x.isNull()) {
                nativeOps.dbClose(x);
            }
            if (y != null && !y.isNull()) {
                nativeOps.dbClose(y);
            }
            if (z != null && !z.isNull()) {
                nativeOps.dbClose(z);
            }
            if (shapeInfo != null && !shapeInfo.isNull()) {
                nativeOps.dbClose(shapeInfo);
            }
        }
    }

    private static void assertLive(OpaqueDataBuffer buffer, String label) {
        assertNotNull(buffer, label + " was null");
        assertTrue(!buffer.isNull(), label + " contained a null native handle");
    }

    private static void assertLive(OpaqueNDArray array, String label) {
        assertNotNull(array, label + " was null");
        assertTrue(!array.isNull(), label + " contained a null native handle");
    }

    private static void assertLive(OpaqueContext context, String label) {
        assertNotNull(context, label + " was null");
        assertTrue(!context.isNull(), label + " contained a null native handle");
    }

    private static boolean isHardwareVulkanDevice(String name) {
        if (name == null || name.trim().isEmpty()) {
            return false;
        }
        String normalized = name.toLowerCase();
        return !normalized.contains("llvmpipe")
                && !normalized.contains("lavapipe")
                && !normalized.contains("swiftshader")
                && !normalized.contains("software");
    }

    private static Nd4jBackend instantiateBackend(String className) {
        try {
            ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
            return Class.forName(className, true, classLoader)
                    .asSubclass(Nd4jBackend.class)
                    .getDeclaredConstructor()
                    .newInstance();
        } catch (ReflectiveOperationException e) {
            throw new AssertionError("Could not initialize backend " + className, e);
        }
    }
}
