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

package org.nd4j.linalg.jcublas;

import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.io.Resource;
import org.nd4j.linalg.util.Paths;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.io.ClassPathResource;

/**
 *
 */
public class JCublasBackend extends Nd4jBackend {

    private static final Logger log = LoggerFactory.getLogger(JCublasBackend.class);

    private final static String LINALG_PROPS = "/nd4j-jcublas.properties";


    @Override
    public boolean isAvailable() {
        // execute SimpleJCublas static initializer to confirm that the library is usable
        try {
            if(!canRun())
                return false;
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
        return true;
    }

    @Override
    public boolean canRun() {
        return Paths.nameExistsInPath("nvcc") || Paths.nameExistsInPath("nvcc.exe");
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


}
