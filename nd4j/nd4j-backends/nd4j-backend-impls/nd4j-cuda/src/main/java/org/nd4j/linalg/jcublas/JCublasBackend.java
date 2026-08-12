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

package org.nd4j.linalg.jcublas;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Loader;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.device.CudaDeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.environment.Nd4jEnvironment;
import org.nd4j.linalg.api.memory.MemoryManager;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.io.Resource;
import org.nd4j.linalg.jcublas.bindings.Nd4jCuda;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import static org.bytedeco.cuda.global.cudart.*;

/**
 *
 */
@Slf4j
public class JCublasBackend extends Nd4jBackend {


    private final static String LINALG_PROPS = "/nd4j-jcublas.properties";


    @Override
    public boolean isAvailable() {
        try {
            if (!canRun())
                return false;
        } catch (Throwable e) {
            while (e.getCause() != null) {
                e = e.getCause();
            }
            log.error("CUDA backend availability check failed", e);
            throw new RuntimeException(e);
        }
        return true;
    }

    @Override
    public boolean canRun() {
        // This is the NVIDIA CUDA backend. ZLUDA has a separate backend and
        // platform classifier, so environment variables must not change this
        // backend's device-selection behavior.
        int[] count = { 0 };
        int errorCode = org.bytedeco.cuda.global.cudart.cudaGetDeviceCount(count);
        if(errorCode != cudaSuccess) {
            System.out.println(cudaGetErrorString(errorCode).getString());
        }

        if (count[0] <= 0) {
            throw new RuntimeException("No CUDA devices were found in system");
        }
        Loader.load(org.bytedeco.cuda.global.cublas.class);

        return true;
    }

    @Override
    public boolean allowsOrder() {
        return false;
    }

    @Override
    public int getPriority() {
        return BACKEND_PRIORITY_GPU;
    }

    @Override
    public Resource getConfigurationResource() {
        return new ClassPathResource(LINALG_PROPS, JCublasBackend.class.getClassLoader());
    }

    @Override
    public Class getNDArrayClass() {
        return JCublasNDArray.class;
    }

    @Override
    public Environment getEnvironment() {
        return CudaEnvironment.getInstance();
    }

    @Override
    public String buildInfo() {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().buildInfo();
    }

    @Override
    public void logBackendInit() {
        String logInitProperty = System.getProperty(ND4JSystemProperties.LOG_INITIALIZATION, "true");
        boolean logInit = Boolean.parseBoolean(logInitProperty);

        if(logInit) {
            try {
                Nd4jCuda.Environment e = Nd4jCuda.Environment.getInstance();
                int blasMajor = e.blasMajorVersion();
                int blasMinor = e.blasMinorVersion();
                int blasPatch = e.blasPatchVersion();
                log.info("ND4J CUDA build version: {}.{}.{}", blasMajor, blasMinor, blasPatch);
                int nGPUs = Nd4jEnvironment.getEnvironment().getNumGpus();

                Properties props = Nd4j.getExecutioner().getEnvironmentInformation();
                List<Map<String, Object>> devicesList = (List<Map<String, Object>>) props.get(Nd4jEnvironment.CUDA_DEVICE_INFORMATION_KEY);

                for (int i = 0; i < nGPUs; i++) {
                    Map<String, Object> dev = devicesList.get(i);
                    String name = (String) dev.get(Nd4jEnvironment.CUDA_DEVICE_NAME_KEY);
                    int major = ((Number) dev.get(Nd4jEnvironment.CUDA_DEVICE_MAJOR_VERSION_KEY)).intValue();
                    int minor = ((Number) dev.get(Nd4jEnvironment.CUDA_DEVICE_MINOR_VERSION_KEY)).intValue();
                    long totalMem = ((Number) dev.get(Nd4jEnvironment.CUDA_TOTAL_MEMORY_KEY)).longValue();
                    log.info("CUDA device {}: [{}]; cc: [{}.{}]; Total memory: [{}]", i, name, major, minor, totalMem);
                }
                log.info("Backend build information:\n {}", buildInfo());
            } catch (Throwable t) {
                log.debug("Error logging CUDA backend versions and devices", t);
            }
        }
    }

    // ============ Multi-Backend Support ============

    @Override
    public List<DeviceDescriptor> discoverDevices() {
        List<DeviceDescriptor> devices = new ArrayList<>();

        try {
            // Backend discovery must use CUDA's own NativeOps authority instead of
            // whichever backend is currently primary in the process.
            Nd4jCuda nativeOps = new Nd4jCuda();
            nativeOps.initializeDevicesAndFunctions();
            int nGPUs = nativeOps.getAvailableDevices();
            long[] totalMemory = new long[nGPUs];
            int bestGpu = 0;
            for (int i = 0; i < nGPUs; i++) {
                totalMemory[i] = nativeOps.getDeviceTotalMemory(i);
                if (totalMemory[i] > totalMemory[bestGpu]) {
                    bestGpu = i;
                }
            }

            for (int i = 0; i < nGPUs; i++) {
                String name = nativeOps.getDeviceName(i);
                int major = nativeOps.getDeviceMajor(i);
                int minor = nativeOps.getDeviceMinor(i);
                long totalMem = totalMemory[i];
                int smCount = estimateSmCount(major, minor, totalMem);
                long memoryBandwidth = estimateMemoryBandwidth(major, minor, totalMem);

                devices.add(new CudaDeviceDescriptor(
                        getBackendId(),
                        i,
                        name,
                        i == bestGpu,
                        major,
                        minor,
                        smCount,
                        totalMem,
                        memoryBandwidth
                ));
            }
        } catch (Exception e) {
            log.warn("Failed to discover CUDA devices from CUDA NativeOps", e);
        }

        return devices;
    }

    private int estimateSmCount(int major, int minor, long totalMem) {
        // Rough estimation based on memory size and architecture
        long memGB = totalMem / (1024L * 1024L * 1024L);
        if (major >= 8) {
            // Ampere: roughly 1 SM per 128MB for high-end
            return Math.max(40, (int) (memGB * 5));
        } else if (major >= 7) {
            // Turing/Volta: roughly 1 SM per 170MB
            return Math.max(30, (int) (memGB * 4));
        }
        return Math.max(16, (int) (memGB * 3));
    }

    private long estimateMemoryBandwidth(int major, int minor, long totalMem) {
        // Bandwidth estimates based on generation
        if (major >= 9) {
            return 2000L * 1024 * 1024 * 1024;  // ~2 TB/s for H100
        } else if (major >= 8) {
            return 1500L * 1024 * 1024 * 1024;  // ~1.5 TB/s for A100
        } else if (major >= 7) {
            if (minor >= 5) {
                return 600L * 1024 * 1024 * 1024;  // ~600 GB/s for RTX 30 series
            }
            return 900L * 1024 * 1024 * 1024;  // ~900 GB/s for V100
        } else if (major >= 6) {
            return 500L * 1024 * 1024 * 1024;  // ~500 GB/s for Pascal
        }
        return 300L * 1024 * 1024 * 1024;  // ~300 GB/s for older
    }

    @Override
    public OpExecutioner createExecutioner() {
        return Nd4j.getExecutioner();
    }

    @Override
    public MemoryManager createMemoryManager() {
        return Nd4j.getMemoryManager();
    }

    @Override
    public String getBackendId() {
        return "cuda";
    }
}
