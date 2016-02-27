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

package org.nd4j.linalg;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestResult;
import org.junit.After;
import org.junit.Before;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.UUID;



/**
 * Base Nd4j test
 * @author Adam Gibson
 */
public abstract class BaseNd4jTest  implements Test {
    private static Logger log = LoggerFactory.getLogger(BaseNd4jTest.class);
    protected Nd4jBackend backend;
    protected String name;
    public final static String DEFAULT_BACKEND = "org.nd4j.linalg.defaultbackend";

    public BaseNd4jTest() {
        this("",getDefaultBackend());
    }

    public BaseNd4jTest(String name) {
       this(name,getDefaultBackend());
    }

    public BaseNd4jTest(String name, Nd4jBackend backend) {
        this.backend = backend;
        this.name = name;
    }

    public BaseNd4jTest(Nd4jBackend backend) {
        this(backend.getClass().getName() + UUID.randomUUID().toString(),backend);

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
        String clazz = System.getProperty(DEFAULT_BACKEND,cpuBackend);
        try {
            return (Nd4jBackend) Class.forName(clazz).newInstance();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }




    @Before
    public void before() {
        log.info("Running " + getClass().getName() + " on backend " + backend.getClass().getName());
        Nd4j nd4j = new Nd4j();
        nd4j.initWithBackend(backend);
        Nd4j.factory().setOrder(ordering());
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;
        Nd4j.MAX_SLICES_TO_PRINT = -1;
    }

    @After
    public void after() {
        log.info("Ending " + getClass().getName());
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




    @Override
    public int countTestCases() {
        return 0;
    }

    @Override
    public void run(TestResult result) {

    }
}
