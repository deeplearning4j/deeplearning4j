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

package org.nd4j.linalg;


import lombok.extern.slf4j.Slf4j;
import org.junit.Before;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.BaseND4JTest;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.*;

import static org.junit.Assume.assumeTrue;


/**
 * Base Nd4j test
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
@Slf4j
public abstract class BaseNd4jTest extends BaseND4JTest {

    protected Nd4jBackend backend;
    protected String name;
    public final static String DEFAULT_BACKEND = "org.nd4j.linalg.defaultbackend";

    public BaseNd4jTest() {
        this("", getDefaultBackend());
    }

    public BaseNd4jTest(String name) {
        this(name, getDefaultBackend());
    }

    public BaseNd4jTest(String name, Nd4jBackend backend) {
        this.backend = backend;
        this.name = name;
    }

    public BaseNd4jTest(Nd4jBackend backend) {
        this(backend.getClass().getName() + UUID.randomUUID().toString(), backend);
    }

    private static List<Nd4jBackend> backends;
    static {
        ServiceLoader<Nd4jBackend> loadedBackends = ServiceLoader.load(Nd4jBackend.class);
        Iterator<Nd4jBackend> backendIterator = loadedBackends.iterator();
        backends = new ArrayList<>();
        List<String> backendsToRun = Nd4jTestSuite.backendsToRun();

        while (backendIterator.hasNext()) {
            Nd4jBackend backend = backendIterator.next();
            if (backend.canRun() && backendsToRun.contains(backend.getClass().getName()) || backendsToRun.isEmpty())
                backends.add(backend);
        }
    }

    @Parameterized.Parameters(name = "{index}: backend({0})={1}")
    public static Collection<Object[]> configs() {
        List<Object[]> ret = new ArrayList<>();
        for (Nd4jBackend backend : backends)
            ret.add(new Object[] {backend});
        return ret;
    }

    @Override
    @Before
    public void beforeTest(){
        super.beforeTest();
        Nd4j.factory().setOrder(ordering());
    }

    /**
     * Get the default backend (jblas)
     * The default backend can be overridden by also passing:
     * -Dorg.nd4j.linalg.defaultbackend=your.backend.classname
     * @return the default backend based on the
     * given command line arguments
     */
    public static Nd4jBackend getDefaultBackend() {
        String cpuBackend = "org.nd4j.linalg.cpu.nativecpu.CpuBackend";
        //String cpuBackend = "org.nd4j.linalg.cpu.CpuBackend";
        String gpuBackend = "org.nd4j.linalg.jcublas.JCublasBackend";
        String clazz = System.getProperty(DEFAULT_BACKEND, cpuBackend);
        try {
            return (Nd4jBackend) Class.forName(clazz).newInstance();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
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

    public String getFailureMessage() {
        return "Failed with backend " + backend.getClass().getName() + " and ordering " + ordering();
    }
}
