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

package org.nd4j.linalg.eigen;

import org.junit.Test;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 9/30/14.
 */
public abstract class EigenTests {

    private static Logger log = LoggerFactory.getLogger(EigenTests.class);


    @Test
    public void testEigen() {
        INDArray linspace = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        IComplexNDArray solution = Nd4j.createComplex(new float[]{-0.37228132f, 0, 0, 0, 0, 0, 5.37228132f, 0}, new int[]{2, 2});
        IComplexNDArray[] eigen = Eigen.eigenvectors(linspace);
        assertEquals(eigen[0], solution);

    }

}
