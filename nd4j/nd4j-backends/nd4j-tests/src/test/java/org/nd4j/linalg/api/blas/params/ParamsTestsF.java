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

package org.nd4j.linalg.api.blas.params;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

/**
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class ParamsTestsF extends BaseNd4jTest {


    public ParamsTestsF(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testGemm() {
        INDArray a = Nd4j.create(2, 2);
        INDArray b = Nd4j.create(2, 3);
        INDArray c = Nd4j.create(2, 3);
        GemmParams params = new GemmParams(a, b, c);
        assertEquals(a.rows(), params.getM());
        assertEquals(b.columns(), params.getN());
        assertEquals(a.columns(), params.getK());
        assertEquals(a.rows(), params.getLda());
        assertEquals(b.rows(), params.getLdb());
        assertEquals(c.rows(), params.getLdc());
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
