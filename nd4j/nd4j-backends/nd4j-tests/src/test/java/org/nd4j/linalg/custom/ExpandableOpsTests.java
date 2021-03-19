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

package org.nd4j.linalg.custom;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ops.compat.CompatStringSplit;
import org.nd4j.linalg.api.ops.util.PrintVariable;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

@Slf4j
public class ExpandableOpsTests extends BaseNd4jTestWithBackends {


    @Override
    public char ordering() {
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCompatStringSplit_1(Nd4jBackend backend) throws Exception {
        val array = Nd4j.create("first string", "second");
        val delimiter = Nd4j.create(" ");

        val exp0 = Nd4j.createFromArray(new long[] {0,0, 0,1, 1,0});
        val exp1 = Nd4j.create("first", "string", "second");

        val results = Nd4j.exec(new CompatStringSplit(array, delimiter));
        assertNotNull(results);
        assertEquals(2, results.length);

        assertEquals(exp0, results[0]);
        assertEquals(exp1, results[1]);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test(Nd4jBackend backend) {
        val arr = Nd4j.createFromArray(0, 1, 2, 3, 4, 5, 6, 7, 8).reshape(3, 3);
        Nd4j.exec(new PrintVariable(arr));

        val row = arr.getRow(1);
        Nd4j.exec(new PrintVariable(row));
    }
}
