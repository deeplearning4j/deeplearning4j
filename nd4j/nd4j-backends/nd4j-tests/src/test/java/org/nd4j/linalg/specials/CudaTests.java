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

package org.nd4j.linalg.specials;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertEquals;

@Slf4j

public class CudaTests extends BaseNd4jTestWithBackends {

    DataType initialType = Nd4j.dataType();


    @BeforeEach
    public void setUp() {
            Nd4j.setDataType(DataType.FLOAT);
        }

    @AfterEach
    public void setDown() {
            Nd4j.setDataType(initialType);
        }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMGrid_1(Nd4jBackend backend) {
        if (!(Nd4j.getExecutioner() instanceof GridExecutioner))
            return;

        val arrayA = Nd4j.create(128, 128);
        val arrayB = Nd4j.create(128, 128);
        val arrayC = Nd4j.create(128, 128);

        arrayA.muli(arrayB);

        val executioner = (GridExecutioner) Nd4j.getExecutioner();

        assertEquals(1, executioner.getQueueLength());

        arrayA.addi(arrayC);

        assertEquals(1, executioner.getQueueLength());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMGrid_2(Nd4jBackend backend) {
        if (!(Nd4j.getExecutioner() instanceof GridExecutioner))
            return;

        val exp = Nd4j.create(128, 128).assign(2.0);
        Nd4j.getExecutioner().commit();

        val arrayA = Nd4j.create(128, 128);
        val arrayB = Nd4j.create(128, 128);
        arrayA.muli(arrayB);

        val executioner = (GridExecutioner) Nd4j.getExecutioner();

        assertEquals(1, executioner.getQueueLength());

        arrayA.addi(2.0f);

        assertEquals(0, executioner.getQueueLength());

        Nd4j.getExecutioner().commit();

        assertEquals(exp, arrayA);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
