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

package org.nd4j.finitedifferences;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.io.ClassPathResource;

import static org.junit.Assert.assertEquals;

public class TwoPointApproximationTest {
    private DataBuffer.Type dtype;

    @Before
    public void setUp() {
        Nd4j.create(1);
        dtype = Nd4j.dataType();

        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ANY_PANIC);
    }

    @After
    public void tearDown() {
        Nd4j.setDataType(dtype);
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.DISABLED);
    }

    @Test
    public void testLinspaceDerivative() throws Exception {

       String basePath = "/two_points_approx_deriv_numpy/";
        INDArray linspace = Nd4j.createNpyFromInputStream(new ClassPathResource(basePath + "x.npy").getInputStream());
        INDArray yLinspace = Nd4j.createNpyFromInputStream(new ClassPathResource(basePath + "y.npy").getInputStream());
        Function<INDArray,INDArray> f = new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray indArray) {
                return indArray.add(1);
            }
        };

        INDArray test = TwoPointApproximation
                .approximateDerivative(f,linspace,null,yLinspace,
                        Nd4j.create(new double[] {Float.MIN_VALUE
                                ,Float.MAX_VALUE}));

        INDArray npLoad = Nd4j.createNpyFromInputStream(new ClassPathResource(basePath + "approx_deriv_small.npy").getInputStream());
        assertEquals(npLoad,test);
        System.out.println(test);

    }
    
}
