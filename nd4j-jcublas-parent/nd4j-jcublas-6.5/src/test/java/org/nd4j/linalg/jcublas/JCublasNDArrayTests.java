/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.jcublas;

import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;


/**
 * NDArrayTests
 *
 * @author Adam Gibson
 */
public class JCublasNDArrayTests extends org.nd4j.linalg.api.test.NDArrayTests {
    private static Logger log = LoggerFactory.getLogger(JCublasNDArrayTests.class);


    @Test
    public void testAddColumn() {
        Nd4j.factory().setOrder('f');
        JCublasNDArray a = (JCublasNDArray) Nd4j.create(new float[]{1, 3, 2, 4, 5, 6}, new int[]{2, 3});
        JCublasNDArray aDup = (JCublasNDArray) Nd4j.create(new float[]{3.0f, 6.0f});
        JCublasNDArray column = (JCublasNDArray) a.getColumn(1);
        column.addi(Nd4j.create(new float[]{1, 2}));

        assertEquals(aDup, column);


    }


}