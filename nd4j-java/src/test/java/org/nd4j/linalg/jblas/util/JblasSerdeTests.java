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

package org.nd4j.linalg.jblas.util;

import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.DataInputStream;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Adam Gibson
 */
public class JblasSerdeTests {
    private static Logger log = LoggerFactory.getLogger(JblasSerdeTests.class);

    @Test
    public void testBinary() throws Exception {
        DoubleMatrix d = new DoubleMatrix();
        ClassPathResource c = new ClassPathResource("/test-matrix.ser");
        d.in(new DataInputStream(c.getInputStream()));
        INDArray assertion = JblasSerde.readJblasBinary(new DataInputStream(c.getInputStream()));
        assertTrue(Arrays.equals(new int[]{d.rows, d.columns}, assertion.shape()));
        for (int i = 0; i < d.rows; i++) {
            for (int j = 0; j < d.columns; j++) {
                assertEquals(d.get(i, j), (double) assertion.getFloat(i, j), 1e-1);
            }
        }
    }

}
