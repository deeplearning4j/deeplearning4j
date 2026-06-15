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
package org.eclipse.deeplearning4j.nd4j.linalg;

import org.eclipse.deeplearning4j.tests.extensions.BackendTest;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;

@BackendTest(description = "Isolation coverage for rank-1 broadcast orientation")
public class Rank1BroadcastIsolationTest {

    @Test
    public void testRank1MulColumnVectorUsesColumnSemanticsInFallbackPath() {
        INDArray matrix = Nd4j.create(DataType.DOUBLE, new long[]{3, 4}, 'f');
        matrix.assign(0.0);
        matrix.getColumn(3).assign(1.0);

        INDArray columnVector = Nd4j.create(new double[]{1, 2, 3});
        INDArray expected = Nd4j.create(new double[][]{
                {0, 0, 0, 1},
                {0, 0, 0, 2},
                {0, 0, 0, 3}
        });

        assertEquals(expected, matrix.mulColumnVector(columnVector));
    }
}
