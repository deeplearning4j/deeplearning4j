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
import org.nd4j.linalg.api.environment.Nd4jEnvironment;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.io.Resource;
import org.nd4j.nativeblas.CudaEnvironment;
import org.nd4j.nativeblas.Nd4jCuda;
import org.nd4j.nativeblas.NativeOpsHolder;
import java.util.List;
import java.util.Map;
import java.util.Properties;

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
            e.printStackTrace();
            throw new RuntimeException(e);
        }
        return true;
    }

    @Override
    public boolean canRun() {
        int[] count = { 0 };
        org.bytedeco.cuda.global.cudart.cudaGetDeviceCount(count);
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
}
