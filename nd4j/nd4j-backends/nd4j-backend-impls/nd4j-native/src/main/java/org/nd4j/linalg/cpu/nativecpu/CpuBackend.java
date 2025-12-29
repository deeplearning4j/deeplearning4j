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

package org.nd4j.linalg.cpu.nativecpu;

import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.io.Resource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CpuBackend extends Nd4jBackend {

    private static final Logger log = LoggerFactory.getLogger(CpuBackend.class);
    private final static String LINALG_PROPS = "/nd4j-native.properties";

    static {
        // CRITICAL: Configure OpenBLAS threading immediately after library loading.
        // This must run before any BLAS operations to prevent TLS corruption crashes.
        initializeOpenBlasThreading();
    }

    /**
     * Initialize OpenBLAS threading configuration.
     * By default, sets OpenBLAS to single-threaded mode to prevent TLS corruption
     * when combined with application-level OpenMP threading.
     * Users can override by setting OPENBLAS_NUM_THREADS or SD_OPENBLAS_THREADS environment variables.
     */
    private static void initializeOpenBlasThreading() {
        try {
            String openblasThreads = System.getenv("OPENBLAS_NUM_THREADS");
            String sdOpenblasThreads = System.getenv("SD_OPENBLAS_THREADS");

            if (openblasThreads == null && sdOpenblasThreads == null) {
                // Neither environment variable is set - configure for single-threaded
                // to prevent TLS corruption crashes with OpenMP
                // Create a temporary instance to call the native method
                new org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu().setOpenBlasThreads(1);
                log.debug("OpenBLAS configured to single-threaded mode to prevent TLS corruption. " +
                        "Set OPENBLAS_NUM_THREADS=N to enable multi-threaded OpenBLAS.");
            }
        } catch (Throwable t) {
            // Ignore errors during initialization - the native library might not be fully loaded yet
            log.debug("Could not configure OpenBLAS threading: {}", t.getMessage());
        }
    }

    @Override
    public boolean isAvailable() {
        return true;
    }

    @Override
    public boolean canRun() {
        //no reliable way (yet!) to determine if running
        return true;
    }

    @Override
    public boolean allowsOrder() {
        return false;
    }

    @Override
    public int getPriority() {
        return BACKEND_PRIORITY_CPU;
    }

    @Override
    public Resource getConfigurationResource() {
        return new ClassPathResource(LINALG_PROPS, CpuBackend.class.getClassLoader());
    }

    @Override
    public Class getNDArrayClass() {
        return NDArray.class;
    }

    @Override
    public Environment getEnvironment() {
        return CpuEnvironment.getInstance();
    }

    @Override
    public String buildInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("PID: " + ProcessHandle.current().pid() + "\n");
        sb.append(Nd4j.getNativeOps().buildInfo());
        return sb.toString();
    }

    @Override
    public void logBackendInit() {
        String logInitProperty = System.getProperty(ND4JSystemProperties.LOG_INITIALIZATION, "true");
        boolean logInit = Boolean.parseBoolean(logInitProperty);

        if(logInit) {
            try {
                log.info("Backend build information:\n {} IP: {}", buildInfo(),ProcessHandle.current().pid());
            } catch (Throwable t) {
                log.debug("Error logging CPU backend ", t);
            }
        }
    }

}

