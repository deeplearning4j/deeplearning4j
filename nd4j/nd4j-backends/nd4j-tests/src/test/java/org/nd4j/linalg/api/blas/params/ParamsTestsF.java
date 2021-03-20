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

package org.nd4j.linalg.api.blas.params;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author Adam Gibson
 */
@NativeTag
public class ParamsTestsF extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGemm (Nd4jBackend backend) {
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
