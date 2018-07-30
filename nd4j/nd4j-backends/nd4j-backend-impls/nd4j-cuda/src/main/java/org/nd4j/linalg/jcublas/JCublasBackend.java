/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.jcublas;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.cublas;
import org.bytedeco.javacpp.cuda;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.io.Resource;
import org.nd4j.linalg.jcublas.complex.JCublasComplexNDArray;

/**
 *
 */
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
            throw new RuntimeException(e);
        }
        return true;
    }

    @Override
    public boolean canRun() {
        int[] count = { 0 };
        cuda.cudaGetDeviceCount(count);
        if (count[0] <= 0) {
            throw new RuntimeException("No CUDA devices were found in system");
        }
        Loader.load(cublas.class);
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
    public Class getComplexNDArrayClass() {
        return JCublasComplexNDArray.class;
    }

}
