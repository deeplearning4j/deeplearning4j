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

package org.nd4j.linalg.jzluda;

import org.bytedeco.javacpp.Loader;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.io.Resource;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.memory.MemoryManager;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.jcublas.bindings.Nd4jCuda;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;

/**
 * ZLUDA backend for ND4J.
 *
 * <p>The platform classifier contains the DL4J CUDA native backend, the pinned
 * ZLUDA CUDA ABI implementation, and its redistributable HIP/ROCm user-space
 * dependency closure. JavaCPP extracts that closure and loads it from one
 * directory. Consumers therefore do not install ZLUDA or set {@code ZLUDA_PATH}
 * or {@code LD_LIBRARY_PATH}. The host still needs a compatible AMD kernel
 * driver and access to the GPU device nodes.</p>
 */
public class JZludaBackend extends Nd4jBackend {

    private static final Logger log = LoggerFactory.getLogger(JZludaBackend.class);
    private static final String LINALG_PROPS = "/nd4j-jzluda.properties";

    public enum ZludaTarget {
        AMD
    }

    private static final ZludaTarget TARGET = ZludaTarget.AMD;
    private final boolean zludaAvailable;
    private final Throwable loadFailure;

    public JZludaBackend() {
        boolean loaded = false;
        Throwable failure = null;
        try {
            // Loading the generated binding makes JavaCPP extract and load the
            // classifier's complete native closure before backend selection.
            Loader.load(Nd4jCuda.class);
            loaded = true;
            log.info("Loaded the bundled ZLUDA/HIP runtime for AMD GPUs");
        } catch (Throwable t) {
            failure = t;
            log.debug("The bundled ZLUDA/HIP runtime could not be loaded", t);
        }
        zludaAvailable = loaded;
        loadFailure = failure;
    }

    @Override
    public boolean isAvailable() {
        return zludaAvailable;
    }

    @Override
    public boolean canRun() {
        // Native loading validates the packaged user-space runtime. Actual device
        // initialization remains in libnd4j and uses ZLUDA's CUDA ABI over HIP;
        // it must never require or probe an NVIDIA CUDA device.
        return zludaAvailable;
    }

    @Override
    public int getPriority() {
        return BACKEND_PRIORITY_GPU - 10;
    }

    @Override
    public Resource getConfigurationResource() {
        return new ClassPathResource(LINALG_PROPS, JZludaBackend.class.getClassLoader());
    }

    @Override
    public Class<?> getNDArrayClass() {
        try {
            return Class.forName("org.nd4j.linalg.jcublas.JCublasNDArray");
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(
                    "Shared CUDA NDArray class not found; "
                            + "nd4j-cuda-backend-common is required",
                    e);
        }
    }

    @Override
    public Environment getEnvironment() {
        return ZludaEnvironment.getInstance();
    }

    public ZludaTarget getTarget() {
        return TARGET;
    }

    public Throwable getLoadFailure() {
        return loadFailure;
    }

    @Override
    public String toString() {
        return "ZLUDA Backend [target=" + TARGET
                + ", bundledRuntime=" + zludaAvailable + "]";
    }

    @Override
    public boolean allowsOrder() {
        return false;
    }

    @Override
    public String buildInfo() {
        StringBuilder builder = new StringBuilder()
                .append("ZLUDA Backend\n")
                .append("Target: ").append(TARGET).append('\n')
                .append("Runtime: bundled JavaCPP platform classifier\n");
        if (loadFailure != null) {
            builder.append("Load failure: ").append(loadFailure).append('\n');
        }
        return builder.toString();
    }

    @Override
    public void logBackendInit() {
        if (Boolean.parseBoolean(System.getProperty(
                ND4JSystemProperties.LOG_INITIALIZATION, "true"))) {
            log.info("ZLUDA Backend build information:\n{}", buildInfo());
        }
    }

    @Override
    public List<DeviceDescriptor> discoverDevices() {
        return Collections.emptyList();
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
        return "zluda";
    }
}
