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

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertEquals;


public class LapackTest extends BaseNd4jTestWithBackends {


    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testQRSquare(Nd4jBackend backend) {
        INDArray A = Nd4j.create(new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9});
        A = A.reshape('c', 3, 3);
        INDArray O = Nd4j.create(A.dataType(), A.shape());
        Nd4j.copy(A, O);
        INDArray R = Nd4j.create(A.dataType(), A.columns(), A.columns());

        Nd4j.getBlasWrapper().lapack().geqrf(A, R);

        A.mmuli(R);
        O.subi(A);
        DataBuffer db = O.data();
        for (int i = 0; i < db.length(); i++) {
            assertEquals(0, db.getFloat(i), 1e-5);
        }
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testQRRect(Nd4jBackend backend) {
        INDArray A = Nd4j.create(new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        A = A.reshape('f', 4, 3);
        INDArray O = Nd4j.create(A.dataType(), A.shape());
        Nd4j.copy(A, O);

        INDArray R = Nd4j.create(A.dataType(), A.columns(), A.columns());
        Nd4j.getBlasWrapper().lapack().geqrf(A, R);

        A.mmuli(R);
        O.subi(A);
        DataBuffer db = O.data();
        for (int i = 0; i < db.length(); i++) {
            assertEquals(0, db.getFloat(i), 1e-5);
        }
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testCholeskyL(Nd4jBackend backend) {
        INDArray A = Nd4j.create(new double[] {2, -1, 1, -1, 2, -1, 1, -1, 2,});
        A = A.reshape('c', 3, 3);
        INDArray O = Nd4j.create(A.dataType(), A.shape());
        Nd4j.copy(A, O);

        Nd4j.getBlasWrapper().lapack().potrf(A, true);

        A.mmuli(A.transpose());
        O.subi(A);
        DataBuffer db = O.data();
        for (int i = 0; i < db.length(); i++) {
            assertEquals(0, db.getFloat(i), 1e-5);
        }
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testCholeskyU(Nd4jBackend backend) {
        INDArray A = Nd4j.create(new double[] {3, -1, 2, -1, 3, -1, 2, -1, 3,});
        A = A.reshape('f', 3, 3);
        INDArray O = Nd4j.create(A.dataType(), A.shape());
        Nd4j.copy(A, O);

        Nd4j.getBlasWrapper().lapack().potrf(A, false);
        A = A.transpose().mmul(A);
        O.subi(A);
        DataBuffer db = O.data();
        for (int i = 0; i < db.length(); i++) {
            assertEquals(0, db.getFloat(i), 1e-5);
        }
    }


    @Override
    public char ordering() {
        return 'f';
    }
}
