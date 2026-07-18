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

package org.nd4j.nativeblas;

import java.io.IOException;
import java.util.Properties;
import java.io.File;
import lombok.Getter;
import lombok.Setter;
import org.bytedeco.javacpp.Loader;
import org.nd4j.common.config.ND4JClassLoading;
import org.nd4j.common.config.ND4JEnvironmentVars;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.io.ReflectionUtils;
import org.nd4j.context.Nd4jContext;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NativeOpsHolder {
    private static Logger log = LoggerFactory.getLogger(NativeOpsHolder.class);
    private static final NativeOpsHolder INSTANCE = new NativeOpsHolder();

    @Getter
    @Setter
    private  NativeOps deviceNativeOps;

    public static int getCores(int totals) {
        // that's special case for Xeon Phi
        if (totals >= 256)
            return 64;

        int ht_off = totals / 2; // we count off HyperThreading without any excuses
        if (ht_off <= 4)
            return 4; // special case for Intel i5. and nobody likes i3 anyway

        if (ht_off > 24) {
            int rounds = 0;
            while (ht_off > 24) { // we loop until final value gets below 24 cores, since that's reasonable
                                  // threshold as of 2016
                if (ht_off > 24) {
                    ht_off /= 2; // we dont' have any cpus that has higher number then 24 physical cores
                    rounds++;
                }
            }
            // 20 threads is special case in this branch
            if (ht_off == 20 && rounds < 2)
                ht_off /= 2;
        } else { // low-core models are known, but there's a gap, between consumer cpus and xeons
            if (ht_off <= 6) {
                // that's more likely consumer-grade cpu, so leave this value alone
                return ht_off;
            } else {
                if (isOdd(ht_off)) // if that's odd number, it's final result
                    return ht_off;

                // 20 threads & 16 threads are special case in this branch, where we go min
                // value
                if (ht_off == 20 || ht_off == 16)
                    ht_off /= 2;
            }
        }
        return ht_off;
    }

    private static boolean isOdd(int value) {
        return (value % 2 != 0);
    }

    private NativeOpsHolder() {
        try {

           if(getDeviceNativeOps() == null && Boolean.parseBoolean(System.getProperty(ND4JSystemProperties.INIT_NATIVEOPS_HOLDER,"true")) ){
               Properties props = Nd4jContext.getInstance().getConf();
               Object nativeOpsDefault = props.get(Nd4j.NATIVE_OPS);
               if(!System.getProperties().containsKey(Nd4j.NATIVE_OPS) && nativeOpsDefault == null) {
                   throw new IllegalStateException("No native operations class found. Please either define native.ops as a system property or fix your relevant backend properties file.");
               }
               String name = System.getProperty(Nd4j.NATIVE_OPS, nativeOpsDefault.toString());
               Class<? extends NativeOps> nativeOpsClass = ND4JClassLoading
                       .loadClassByName(name)
                       .asSubclass(NativeOps.class);
               DeviceType deviceType = deviceTypeFromClassName(name);
               MultiBackendNativeOpsHolder multiBackendHolder =
                       MultiBackendNativeOpsHolder.getInstance();

               // Multi-backend discovery can legitimately run before the legacy holder
               // is first requested (for example while a backend-scoped runtime is
               // being assembled). Reuse that exact initialized binding instead of
               // constructing a second JavaCPP authority for the same native backend.
               synchronized (multiBackendHolder) {
                   NativeOps registeredOps =
                           multiBackendHolder.getLoadedBackendIfPresent(deviceType);
                   if (registeredOps != null) {
                       if (!nativeOpsClass.isInstance(registeredOps)) {
                           throw new IllegalStateException(
                                   "The registered " + deviceType + " NativeOps binding is "
                                           + registeredOps.getClass().getName()
                                           + ", but the active backend requested " + name);
                       }
                       deviceNativeOps = registeredOps;
                       initOps(false);
                       log.debug("Reusing multi-backend NativeOps binding for {}", deviceType);
                   } else {
                       deviceNativeOps = ReflectionUtils.newInstance(nativeOpsClass);
                       initOps(true);

                       // Register this NativeOps so later multi-backend discovery reuses
                       // the exact active binding.
                       registerWithMultiBackendHolder(name);
                   }
               }

               // Load any native plugin libraries from the plugin directory.
               // This must happen AFTER the main native library is initialized,
               // since plugins may depend on symbols from the primary backend.
               NativePluginLoader.loadPlugins();

               // Register shutdown hook to set the shutdown flag EARLY.
               // This prevents SIGSEGV crashes if any code accidentally calls
               // clearTADCache() during JVM shutdown (e.g., finalizers, GC, or other hooks).
               //
               // NOTE: We ONLY set the shutdown flag here - we do NOT clear the caches!
               // During JVM shutdown, memory allocators may have been destroyed,
               // causing corrupted pointers in the cache tries. The OS will reclaim
               // all memory when the process exits anyway.
               //
               // For explicit cleanup during runtime (e.g., testing), call:
               //   Nd4j.clearTADCache() and Nd4j.clearShapeCache() directly.
               Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                   try {
                       // Set the plan cache shutdown flag FIRST — this prevents unpinPlan()
                       // and LRU eviction from calling cudaStreamDestroy/cudaFree/cudaGraphExecDestroy
                       // while the CUDA context is being torn down (SIGSEGV fix).
                       deviceNativeOps.setPlanCacheShutdownInProgress(true);
                       // Set the shutdown flags - this makes clearTADCache() and clearShapeCache()
                       // safe to call (they will return early without traversing potentially corrupted memory)
                       deviceNativeOps.setTADCacheShutdownInProgress(true);
                       deviceNativeOps.setShapeCacheShutdownInProgress(true);
                       // Stop the DeallocatorService from calling native free() during shutdown.
                       // Heap metadata may be corrupted from prior buffer overruns, and calling
                       // free() triggers SIGABRT. The OS reclaims all memory on process exit.
                       org.nd4j.linalg.api.memory.deallocation.DeallocatorService.getShutdownInProgress().set(true);
                   } catch (Throwable t) {
                       // Ignore errors during shutdown - we're just trying to be safe
                   }
               }, "ND4J-Shutdown-Flag"));

           }


        } catch (Exception | Error e) {
            throw new RuntimeException(
                    "ND4J is probably missing dependencies. For more information, please refer to: https://deeplearning4j.konduit.ai/nd4j/backend",
                    e);
        }

        // Fail fast when an accelerator backend is active without a usable device. Backend
        // selection must reflect actual runtime capability before the first native operation.
        verifyUsableDeviceOrFail();
    }

    public void initOps() {
        initOps(true);
    }

    private void initOps(boolean initializeDevicesAndFunctions) {
        if (initializeDevicesAndFunctions) {
            deviceNativeOps.initializeDevicesAndFunctions();
        }

        // Removed explicit cache initialization - caches use lazy singleton initialization
        // Previous attempts to pre-initialize caused issues with initialization tracking atomics
        // The shape and TAD caches will initialize naturally on first use via getInstance()

        int numThreads;
        String numThreadsString = System.getenv(ND4JEnvironmentVars.OMP_NUM_THREADS);
        if (numThreadsString != null && !numThreadsString.isEmpty()) {
            numThreads = Integer.parseInt(numThreadsString);
            deviceNativeOps.setOmpNumThreads(numThreads);
        } else {
            int cores = Loader.totalCores();
            int chips = Loader.totalChips();
            if (chips > 0 && cores > 0) {
                deviceNativeOps.setOmpNumThreads(Math.max(1, cores / chips));
            } else
                deviceNativeOps.setOmpNumThreads(
                        getCores(Runtime.getRuntime().availableProcessors()));
        }

        // OpenBLAS threads are NOT configured here. OpenBLAS follows the single max-threads
        // setting applied by the native config authority (CoreConfig::initFromEnvironment,
        // driven by _maxThreads / OMP_NUM_THREADS / SD_MAX_THREADS). There is no
        // backend-specific OpenBLAS knob to set from Java.

        // Configure BLAS call serialization.
        // Only push a value down when the user EXPLICITLY set ND4J_BLAS_SERIALIZE —
        // unconditionally forcing true here stomped the native-side default at every
        // JVM init, making BlasHelper's BLAS-aware default (and its SD_BLAS_SERIALIZE
        // env fallback) dead code. Native default: MKL (documented thread-safe) runs
        // unserialized; OpenBLAS keeps the safety mutex. ND4J_BLAS_SERIALIZE still
        // wins in both directions when set.
        String serializeBlasString = System.getenv(ND4JEnvironmentVars.ND4J_BLAS_SERIALIZE);
        if (serializeBlasString != null && !serializeBlasString.isEmpty()) {
            String val = serializeBlasString.toLowerCase().trim();
            boolean serializeBlasCalls = !(val.equals("false") || val.equals("0") || val.equals("no"));
            deviceNativeOps.setSerializeBlasCalls(serializeBlasCalls);
        }

        // Propagate DSP execution mode from -Dnd4j.dsp.executionMode into libnd4j.
        // Controls slot-by-slot lifecycle gating: whether actuality ticks, host/device
        // resyncs, and orphan closes run on buffers owned by an in-flight DSP plan.
        // Values: LEGACY_UNAWARE | COEXIST_SAFE (default) | STRICT_ISOLATED, or 0/1/2.
        String dspModeProp = System.getProperty(ND4JSystemProperties.DSP_EXECUTION_MODE);
        if (dspModeProp != null && !dspModeProp.isEmpty()) {
            int dspModeValue;
            String trimmed = dspModeProp.trim();
            try {
                dspModeValue = Integer.parseInt(trimmed);
            } catch (NumberFormatException ex) {
                String upper = trimmed.toUpperCase();
                if (upper.equals("LEGACY_UNAWARE") || upper.equals("LEGACY")) {
                    dspModeValue = 0;
                } else if (upper.equals("STRICT_ISOLATED") || upper.equals("STRICT")) {
                    dspModeValue = 2;
                } else {
                    dspModeValue = 1;
                }
            }
            deviceNativeOps.dspSetExecutionMode(dspModeValue);
            if (log.isDebugEnabled()) {
                log.debug("DSP execution mode set to {} ({})", dspModeValue, dspModeProp);
            }
        }

        String logInitProperty = System.getProperty(ND4JSystemProperties.LOG_INITIALIZATION, "true");
        boolean logInit = Boolean.parseBoolean(logInitProperty);

        try {

            //extract vednn in either graalvm or java
            String vednnUrl = "org/nd4j/linalg/cpu/nativecpu/bindings/linux-x86_64-vednn-avx2/libnd4jcpu_device.vso";
            String vednnUrlGraal = "linux-x86_64-vednn-avx2/libnd4jcpu_device.vso";

            String vednnUrlStatic = "org/nd4j/linalg/cpu/nativecpu/bindings/linux-x86_64-vednn-avx2/libnd4jcpu_device.vsa";
            String vednnUrlGraalStatic = "linux-x86_64-vednn-avx2/libnd4jcpu_device.vsa";

            for(String url : new String[]{vednnUrl,vednnUrlGraal,vednnUrlStatic,vednnUrlGraalStatic}) {
                extractVeIfNeeded(logInit, url);

            }


        } catch (java.io.IOException ignored) {
            // VE library extraction is optional — not all platforms have it
        }

        if (logInit) {
            log.info("Number of threads used for linear algebra (incl. OpenBLAS): {}", deviceNativeOps.ompGetMaxThreads());
            String serializeBlasEnv = System.getenv(ND4JEnvironmentVars.ND4J_BLAS_SERIALIZE);
            log.info("BLAS call serialization: {} (native BLAS-aware default unless ND4J_BLAS_SERIALIZE is set)",
                    (serializeBlasEnv == null || serializeBlasEnv.isEmpty()) ? "native default" : serializeBlasEnv);
        }

        // DISABLED: Custom signal handlers interfere with JVM's crash handling chain
        // The custom handlers were causing double-crashes: when the primary crash occurs,
        // the signal handler tries to chain to JVM's handler, but the handler address
        // becomes invalid, causing a second crash in PosixSignals::chained_handler().
        //
        // Root cause: Installing custom signal handlers on top of JVM's signal handlers
        // corrupts the signal handler chain. When a crash occurs, the chained handler
        // address is invalid (0x00007ff1cf625955 in crash dump), causing SIGSEGV in
        // libjvm.so itself.
        //
        // Solution: Let JVM handle all crashes natively. JVM's crash handling already
        // generates hs_err files and properly handles signals. Custom handlers are not needed.
        //
        // See: contrib/benchmarking_nd4j/hs_err_pid950199.log lines 9, 33-41 for crash details.
        //
        // deviceNativeOps.initializeLifecycleCrashHandlers();  // DISABLED
    }

    private void extractVeIfNeeded(boolean logInit, String vednnUrl) throws IOException {
        ClassPathResource vednnResource = new ClassPathResource(vednnUrl);
        if(vednnResource.exists()) {
            File file = Loader.cacheResource(vednnUrl);
            if (file != null) {
                String path = file.getAbsoluteFile().getParent();
                if (logInit) {
                    log.info("Veda device library cache path: {}", path);
                }

            }
        }
    }

    public static NativeOpsHolder getInstance() {
        return INSTANCE;
    }

    /**
     * Register the loaded NativeOps with MultiBackendNativeOpsHolder.
     * This allows MultiBackendNativeOpsHolder to reuse this instance instead of
     * loading the same backend again.
     *
     * @param className the class name of the NativeOps implementation
     */
    private void registerWithMultiBackendHolder(String className) {
        DeviceType deviceType = deviceTypeFromClassName(className);
        if (deviceType != null && deviceNativeOps != null) {
            MultiBackendNativeOpsHolder.getInstance().registerPreloadedBackend(deviceType, deviceNativeOps);
            log.debug("Registered primary backend {} with MultiBackendNativeOpsHolder", deviceType);
        }
    }

    /**
     * Best-effort mapping from a NativeOps implementation class name to its {@link DeviceType}.
     *
     * @param className fully-qualified NativeOps class name
     * @return the device type, or null if it cannot be classified
     */
    private static DeviceType deviceTypeFromClassName(String className) {
        if (className == null) {
            return null;
        }
        if (className.contains("Nd4jCpu") || className.contains("cpu")) {
            return DeviceType.CPU;
        } else if (className.contains("Nd4jCuda") || className.contains("cuda") || className.contains("jcublas")) {
            return DeviceType.CUDA_GPU;
        } else if (className.contains("Nd4jVulkan") || className.contains("vulkan")) {
            return DeviceType.VULKAN_GPU;
        } else if (className.contains("Nd4jZluda") || className.contains("zluda")) {
            return DeviceType.ROCM_GPU;
        } else if (className.contains("Nd4jTpu") || className.contains("tpu")) {
            return DeviceType.TPU;
        } else if (className.contains("Nd4jMetal") || className.contains("metal")) {
            return DeviceType.METAL_GPU;
        }
        return null;
    }

    /**
     * Verify that an accelerator backend committed as the active backend exposes at least one
     * usable device. If it does not, throw an {@link IllegalStateException} before native
     * operations execute against an invalid runtime.
     *
     * <p>This sits at the single point where the selected NativeOps is instantiated, so it covers
     * every selection path (ServiceLoader load, {@code -Dnative.ops} override, forced backend).
     * The CPU backend is always exempt: it has a logical device and may report 0.</p>
     */
    private void verifyUsableDeviceOrFail() {
        NativeOps ops = deviceNativeOps;
        if (ops == null) {
            return; // initialization skipped (INIT_NATIVEOPS_HOLDER=false) or externally provided
        }

        DeviceType deviceType = deviceTypeFromClassName(ops.getClass().getName());
        // Generic gate: any classified non-CPU (accelerator) backend — CUDA, ROCm/ZLUDA, Metal,
        // TPU, or any future backend — needs at least one usable device. CPU is exempt (it has a
        // logical device and may report 0); an unclassifiable backend (null) is left alone.
        boolean acceleratorBackend = deviceType != null && deviceType != DeviceType.CPU;
        if (!acceleratorBackend) {
            return;
        }

        int devices;
        try {
            devices = ops.getAvailableDevices();
        } catch (Throwable t) {
            // A throw from the device probe means the runtime itself is unusable.
            devices = 0;
        }

        if (devices <= 0) {
            throw new IllegalStateException(
                    "The " + deviceType + " backend (" + ops.getClass().getName() + ") was selected as the "
                    + "active ND4J backend, but its runtime reported no usable device "
                    + "(getAvailableDevices()=" + devices + ").");
        }
    }
}
