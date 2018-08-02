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

package org.nd4j.linalg.api.blas;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.api.buffer.DataBuffer;

import static org.junit.Assert.assertEquals;

/**
 * @author rcorbish
 */
@RunWith(Parameterized.class)
public class LapackTest extends BaseNd4jTest {
    public LapackTest(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testQRSquare() {
        INDArray A = Nd4j.create(new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9});
        A = A.reshape('c', 3, 3);
        INDArray O = Nd4j.create(A.shape());
        Nd4j.copy(A, O);
        INDArray R = Nd4j.create(A.columns(), A.columns());

        Nd4j.getBlasWrapper().lapack().geqrf(A, R);

        A.mmuli(R);
        O.subi(A);
        DataBuffer db = O.data();
        for (int i = 0; i < db.length(); i++) {
            assertEquals(0, db.getFloat(i), 1e-5);
        }
    }

    @Test
    public void testQRRect() {
        INDArray A = Nd4j.create(new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        A = A.reshape('f', 4, 3);
        INDArray O = Nd4j.create(A.shape());
        Nd4j.copy(A, O);

        INDArray R = Nd4j.create(A.columns(), A.columns());
        Nd4j.getBlasWrapper().lapack().geqrf(A, R);

        A.mmuli(R);
        O.subi(A);
        DataBuffer db = O.data();
        for (int i = 0; i < db.length(); i++) {
            assertEquals(0, db.getFloat(i), 1e-5);
        }
    }

    @Test
    public void testCholeskyL() {
        INDArray A = Nd4j.create(new double[] {2, -1, 1, -1, 2, -1, 1, -1, 2,});
        A = A.reshape('c', 3, 3);
        INDArray O = Nd4j.create(A.shape());
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
    public void testCholeskyU() {
        INDArray A = Nd4j.create(new double[] {2, -1, 2, -1, 2, -1, 2, -1, 2,});
        A = A.reshape('f', 3, 3);
        INDArray O = Nd4j.create(A.shape());
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
