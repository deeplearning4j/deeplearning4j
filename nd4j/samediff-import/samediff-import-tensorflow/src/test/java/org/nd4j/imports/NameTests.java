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

package org.nd4j.imports;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertEquals;

@Slf4j

public class NameTests  extends BaseNd4jTestWithBackends {


    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testNameExtraction_1(Nd4jBackend backend) {
        val str = "Name";
        val exp = "Name";

        val pair = SameDiff.parseVariable(str);
        assertEquals(exp, pair.getFirst());
        assertEquals(0, pair.getSecond().intValue());
    }


    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testNameExtraction_2(Nd4jBackend backend) {
        val str = "Name_2";
        val exp = "Name_2";

        val pair = SameDiff.parseVariable(str);
        assertEquals(exp, pair.getFirst());
        assertEquals(0, pair.getSecond().intValue());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testNameExtraction_3(Nd4jBackend backend) {
        val str = "Name_1:2";
        val exp = "Name_1";

        val pair = SameDiff.parseVariable(str);
        assertEquals(exp, pair.getFirst());
        assertEquals(2, pair.getSecond().intValue());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testNameExtraction_4(Nd4jBackend backend) {
        val str = "Name_1:1:2";
        val exp = "Name_1:1";

        val pair = SameDiff.parseVariable(str);
        assertEquals(exp, pair.getFirst());
        assertEquals(2, pair.getSecond().intValue());
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
