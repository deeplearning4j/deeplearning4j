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


package org.nd4j.linalg.aurora;

import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.io.Resource;
import org.nd4j.nativeblas.NativeOpsHolder;

/**
 * Aurora backend
 *
 * @author Adam Gibson
 */
public class AuroraBackend extends Nd4jBackend {


    private final static String LINALG_PROPS = "/nd4j-aurora.properties";

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
        return BACKEND_PRIORITY_GPU;
    }

    @Override
    public Resource getConfigurationResource() {
        return new ClassPathResource(LINALG_PROPS, AuroraBackend.class.getClassLoader());
    }

    @Override
    public Class getNDArrayClass() {
        return AuroraNDArray.class;
    }

    @Override
    public Environment getEnvironment() {
        return AuroraEnvironment.getInstance();
    }

    @Override
    public String buildInfo() {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().buildInfo();
    }

    @Override
    public void logBackendInit() {
        //No additional logging for CPU backend
    }
}
