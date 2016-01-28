/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.cpu;

import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.io.Resource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.io.ClassPathResource;

/**
 * Jblas backend
 *
 */
public class CpuBackend extends Nd4jBackend {

    private static final Logger log = LoggerFactory.getLogger(CpuBackend.class);

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
    public int getPriority() {
        return BACKEND_PRIORITY_CPU;
    }

    @Override
    public Resource getConfigurationResource() {
        return new ClassPathResource(LINALG_PROPS, CpuBackend.class.getClassLoader());
    }
}
