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

package org.nd4j.linalg.cpu.nativecpu;

import org.nd4j.linalg.cpu.nativecpu.complex.ComplexNDArray;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.io.Resource;

/**
 * Cpu backend
 *
 * @author Adam Gibson
 */
public class CpuBackend extends Nd4jBackend {


    private final static String LINALG_PROPS = "/nd4j-native.properties";

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
    public Class getComplexNDArrayClass() {
        return ComplexNDArray.class;
    }
}
