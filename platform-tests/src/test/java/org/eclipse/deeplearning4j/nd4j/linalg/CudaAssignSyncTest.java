/*
 *  ******************************************************************************
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

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CudaAssignSyncTest {

    private static INDArray exactHalfValues() {
        // IEEE fp16 represents every integer in [0, 2048] exactly.
        return Nd4j.linspace(1.0, 2048.0, 2048, DataType.FLOAT).reshape(64, 32);
    }

    @Test
    public void testCastToHalfPreservesContiguousValues() {
        INDArray src = exactHalfValues();
        INDArray cast = src.castTo(DataType.HALF);
        INDArray roundTrip = cast.castTo(DataType.FLOAT);

        double maxDiff = Transforms.abs(src.sub(roundTrip), false).maxNumber().doubleValue();
        assertTrue(maxDiff <= 1e-3, "FLOAT->HALF->FLOAT round-trip drifted by " + maxDiff);
    }

    @Test
    public void testMixedDtypeAssignPreservesContiguousValues() {
        INDArray src = exactHalfValues();
        INDArray dst = Nd4j.create(DataType.HALF, 64, 32);

        dst.assign(src);

        INDArray roundTrip = dst.castTo(DataType.FLOAT);
        double maxDiff = Transforms.abs(src.sub(roundTrip), false).maxNumber().doubleValue();
        assertTrue(maxDiff <= 1e-3, "FLOAT->HALF assign drifted by " + maxDiff);
    }

    @Test
    public void testMixedDtypeScalarAssignPreservesValue() {
        INDArray src = Nd4j.scalar(DataType.FLOAT, 1.25f);
        INDArray dst = Nd4j.scalar(DataType.HALF, 0.0f);

        dst.assign(src);

        assertEquals(1.25f, dst.getFloat(0), 1e-3f);
    }
}
