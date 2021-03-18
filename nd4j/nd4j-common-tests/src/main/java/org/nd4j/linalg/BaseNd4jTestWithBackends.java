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

package org.nd4j.linalg;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.runners.Parameterized;
import org.nd4j.common.config.ND4JClassLoading;
import org.nd4j.common.io.ReflectionUtils;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.*;
import java.util.stream.Stream;

/**
 * Base Nd4j test
 * @author Adam Gibson
 */

@Slf4j
public abstract class BaseNd4jTestWithBackends extends BaseND4JTest {
    private static List<Nd4jBackend> BACKENDS = new ArrayList<>();
    static {
        List<String> backendsToRun = Nd4jTestSuite.backendsToRun();

        ServiceLoader<Nd4jBackend> loadedBackends = ND4JClassLoading.loadService(Nd4jBackend.class);
        for (Nd4jBackend backend : loadedBackends) {
            if (backend.canRun() && backendsToRun.contains(backend.getClass().getName())
                    || backendsToRun.isEmpty()) {
                BACKENDS.add(backend);
            }
        }
    }

    public final static String DEFAULT_BACKEND = "org.nd4j.linalg.defaultbackend";



    public static Stream<Arguments> configs() {
        Stream<Arguments> ret =  BACKENDS.stream().map(input -> Arguments.of(input));
        return ret;
    }

    @BeforeEach
    public void beforeTest2(){
        Nd4j.factory().setOrder(ordering());
    }

    /**
     * Get the default backend (nd4j)
     * The default backend can be overridden by also passing:
     * -Dorg.nd4j.linalg.defaultbackend=your.backend.classname
     * @return the default backend based on the
     * given command line arguments
     */
    public static Nd4jBackend getDefaultBackend() {
        String cpuBackend = "org.nd4j.linalg.cpu.nativecpu.CpuBackend";
        String defaultBackendClass = System.getProperty(DEFAULT_BACKEND, cpuBackend);

        Class<Nd4jBackend> backendClass = ND4JClassLoading.loadClassByName(defaultBackendClass);
        return ReflectionUtils.newInstance(backendClass);
    }

    /**
     * The ordering for this test
     * This test will only be invoked for
     * the given test  and ignored for others
     *
     * @return the ordering for this test
     */
    public char ordering() {
        return 'c';
    }

    public String getFailureMessage(Nd4jBackend backend) {
        return "Failed with backend " + backend.getClass().getName() + " and ordering " + ordering();
    }
}
