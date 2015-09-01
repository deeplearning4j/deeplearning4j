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

package org.nd4j.linalg.jblas;

import org.nd4j.linalg.factory.Nd4jBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;

/**
 * Jblas backend
 *
 */
public class JblasBackend extends Nd4jBackend {

    private static final Logger log = LoggerFactory.getLogger(JblasBackend.class);

    private final static String LINALG_PROPS = "/nd4j-jblas.properties";

    @Override
    public boolean isAvailable() {
        // execute JBLAS static initializer to confirm that the library is usable
        try {
            Class.forName("org.jblas.NativeBlas");
        } catch (Throwable e) {
            log.warn("unable to load Jblas backend", e);
            return false;
        }
        return true;
    }

    @Override
    public boolean canRun() {
        //no reliable way (yet!) to determine if running
        return true;
    }

    @Override
    public int getPriority() {
        return BACKEND_PRIORITY_CPU-1;
    }

    @Override
    public Resource getConfigurationResource() {
        return new ClassPathResource(LINALG_PROPS, JblasBackend.class.getClassLoader());
    }
}
