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

package org.nd4j.linalg.lapack;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertEquals;

@Slf4j

public class LapackTestsF extends BaseNd4jTestWithBackends {
    DataType initialType = Nd4j.dataType();

    @BeforeEach
    public void setUp() {
        Nd4j.setDataType(DataType.DOUBLE);
    }

    @AfterEach
    public void after() {
        Nd4j.setDataType(initialType);
    }

      @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testGetRF1DifferentOrders(Nd4jBackend backend) {
        INDArray a = Nd4j.create(new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9}, new int[] {3, 3}, 'c').dup('f');
        INDArray exp = Nd4j.create(new double[] {7.0, 8.0, 9.0, 0.14285715, 0.85714287, 1.7142857, 0.5714286, 0.5, 0.0},
                        new int[] {3, 3}, 'c').dup('f');

        INDArray r = Nd4j.getNDArrayFactory().lapack().getrf(a);

        assertEquals(exp, a);
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
