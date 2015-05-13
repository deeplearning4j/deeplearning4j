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

import junit.framework.TestCase;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.LinearViewNDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.jblas.JblasBackend;

import java.util.UUID;

/**
 * Base Nd4j test
 * @author Adam Gibson
 */
public abstract class BaseNd4jTest extends TestCase {
    protected Nd4jBackend backend;

    public BaseNd4jTest() {
        this("",new JblasBackend());
    }

    public BaseNd4jTest(String name) {
       this(name,new JblasBackend());
    }

    public BaseNd4jTest(String name, Nd4jBackend backend) {
        super(name);
        this.backend = backend;
    }

    public BaseNd4jTest(Nd4jBackend backend) {
        this(backend.getClass().getName() + UUID.randomUUID().toString(),backend);

    }

    @Before
    public void before() {
        Nd4j nd4j = new Nd4j();
        nd4j.initWithBackend(backend);
        Nd4j.factory().setOrder(ordering());
    }

    @After
    public void after() {
        Nd4j nd4j = new Nd4j();
        nd4j.initWithBackend(backend);
        Nd4j.factory().setOrder(ordering());
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




    @Test
    public void testNewLinearView() {
        INDArray arange = Nd4j.arange(1,17).reshape(4, 4);
        NDArrayIndex index = NDArrayIndex.interval(0, 2);
        INDArray get = arange.get(index, index);
        LinearViewNDArray linearViewNDArray = new LinearViewNDArray(get);
        assertEquals(Nd4j.create(new double[]{1,5,2,6}),linearViewNDArray);

    }




    @Override
    public String getName() {
        return getClass().getName();
    }


}
