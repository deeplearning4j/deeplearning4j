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


import org.junit.After;
import org.junit.Before;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;


/**
 * Base Nd4j test
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public abstract class BaseNd4jTest {
    private static Logger log = LoggerFactory.getLogger(BaseNd4jTest.class);
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

        //Suppress ND4J initialization - don't need this logged for every test...
        System.setProperty(Nd4j.LOG_INIT_ENV_PROPERTY, "false");
        System.gc();
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
    public static void assertArrayEquals(String string, Object[] expecteds, Object[] actuals) {
        org.junit.Assert.assertArrayEquals(string, expecteds, actuals);
    }

    public static void assertArrayEquals(Object[] expecteds, Object[] actuals) {
        org.junit.Assert.assertArrayEquals(expecteds, actuals);
    }

    public static void assertArrayEquals(String string, long[] shapeA, long[] shapeB) {
        org.junit.Assert.assertArrayEquals(string, shapeA, shapeB);
    }

    public static void assertArrayEquals(long[] shapeA, long[] shapeB) {
        org.junit.Assert.assertArrayEquals(shapeA, shapeB);
    }

    public static void assertArrayEquals(String string, int[] shapeA, long[] shapeB) {
        org.junit.Assert.assertArrayEquals(string, ArrayUtil.toLongArray(shapeA), shapeB);
    }

    public static void assertArrayEquals(int[] shapeA, long[] shapeB) {
        org.junit.Assert.assertArrayEquals(ArrayUtil.toLongArray(shapeA), shapeB);
    }

    public static void assertArrayEquals(String string, long[] shapeA, int[] shapeB) {
        org.junit.Assert.assertArrayEquals(string, shapeA, ArrayUtil.toLongArray(shapeB));
    }

    public static void assertArrayEquals(long[] shapeA, int[] shapeB) {
        org.junit.Assert.assertArrayEquals(shapeA, ArrayUtil.toLongArray(shapeB));
    }

    public static void assertArrayEquals(String string, int[] shapeA, int[] shapeB) {
        org.junit.Assert.assertArrayEquals(string, shapeA, shapeB);
    }

    public static void assertArrayEquals(int[] shapeA, int[] shapeB) {
        org.junit.Assert.assertArrayEquals(shapeA, shapeB);
    }

    public static void assertArrayEquals(String string, boolean[] shapeA, boolean[] shapeB) {
        org.junit.Assert.assertArrayEquals(string, shapeA, shapeB);
    }

    public static void assertArrayEquals(boolean[] shapeA, boolean[] shapeB) {
        org.junit.Assert.assertArrayEquals(shapeA, shapeB);
    }


    public static void assertArrayEquals(float[] shapeA, float[] shapeB, float delta) {
        org.junit.Assert.assertArrayEquals(shapeA, shapeB, delta);
    }

    public static void assertArrayEquals(double[] shapeA, double[] shapeB, double delta) {
        org.junit.Assert.assertArrayEquals(shapeA, shapeB, delta);
    }

    public static void assertArrayEquals(String string, float[] shapeA, float[] shapeB, float delta) {
        org.junit.Assert.assertArrayEquals(string, shapeA, shapeB, delta);
    }

    public static void assertArrayEquals(String string, double[] shapeA, double[] shapeB, double delta) {
        org.junit.Assert.assertArrayEquals(string, shapeA, shapeB, delta);
    }

    @Parameterized.Parameters(name = "{index}: backend({0})={1}")
    public static Collection<Object[]> configs() {
        List<Object[]> ret = new ArrayList<>();
        for (Nd4jBackend backend : backends)
            ret.add(new Object[] {backend});
        return ret;
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



    @Before
    public void before() throws Exception {
        log.info("Running " + getClass().getName() + " on backend " + backend.getClass().getName());
        Nd4j nd4j = new Nd4j();
        nd4j.initWithBackend(backend);
        Nd4j.factory().setOrder(ordering());
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;
        Nd4j.MAX_SLICES_TO_PRINT = -1;
    }

    @After
    public void after() throws Exception {
        log.info("Ending " + getClass().getName());
        if (System.getProperties().getProperty("backends") != null
                        && !System.getProperty("backends").contains(backend.getClass().getName()))
            return;
        Nd4j nd4j = new Nd4j();
        nd4j.initWithBackend(backend);
        Nd4j.factory().setOrder(ordering());
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;
        Nd4j.MAX_SLICES_TO_PRINT = -1;
    }


    /**
     * The ordering for this test
     * This test will only be invoked for
     * the given test  and ignored for others
     *
     * @return the ordering for this test
     */
    public char ordering() {
        return 'a';
    }



    public String getFailureMessage() {
        return "Failed with backend " + backend.getClass().getName() + " and ordering " + ordering();
    }



}
