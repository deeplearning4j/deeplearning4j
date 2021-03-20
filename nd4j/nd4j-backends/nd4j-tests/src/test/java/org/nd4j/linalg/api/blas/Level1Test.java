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

package org.nd4j.linalg.api.blas;

import lombok.val;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author Adam Gibson
 */
@NativeTag
public class Level1Test extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDot(Nd4jBackend backend) {
        INDArray vec1 = Nd4j.create(new float[] {1, 2, 3, 4});
        INDArray vec2 = Nd4j.create(new float[] {1, 2, 3, 4});
        assertEquals(30, Nd4j.getBlasWrapper().dot(vec1, vec2), 1e-1);

        INDArray matrix = Nd4j.linspace(1, 4, 4, DataType.FLOAT).reshape(2, 2);
        INDArray row = matrix.getRow(1);
        double dot = Nd4j.getBlasWrapper().dot(row, row);
        assertEquals(20, dot, 1e-1);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAxpy(Nd4jBackend backend) {
        INDArray matrix = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray row = matrix.getRow(1);
        Nd4j.getBlasWrapper().level1().axpy(row.length(), 1.0, row, row);
        assertEquals(Nd4j.create(new double[] {4, 8}), row,getFailureMessage(backend));

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAxpy2(Nd4jBackend backend) {
        val rowX = Nd4j.create(new double[]{1, 2, 3, 4});
        val rowY = Nd4j.create(new double[]{1, 2, 3, 4});
        val exp = Nd4j.create(new double[]{3, 6, 9, 12});

        val z = Nd4j.getBlasWrapper().axpy(2.0, rowX, rowY);
        assertEquals(exp, z);
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
