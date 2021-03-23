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
package org.deeplearning4j.nn.adapters;

import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.util.ArrayUtil;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;

@DisplayName("Regression 2 d Adapter Test")
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
class Regression2dAdapterTest extends BaseDL4JTest {

    @Test
    @DisplayName("Test Regression Adapter _ 2 D _ 1")
    void testRegressionAdapter_2D_1() throws Exception {
        val in = new double[][] { { 1, 2, 3 }, { 4, 5, 6 } };
        val adapter = new Regression2dAdapter();
        val result = adapter.apply(Nd4j.create(in));
        assertArrayEquals(ArrayUtil.flatten(in), ArrayUtil.flatten(result), 1e-5);
    }

    @Test
    @DisplayName("Test Regression Adapter _ 2 D _ 2")
    void testRegressionAdapter_2D_2() throws Exception {
        val in = new double[] { 1, 2, 3 };
        val adapter = new Regression2dAdapter();
        val result = adapter.apply(Nd4j.create(in));
        assertArrayEquals(in, ArrayUtil.flatten(result), 1e-5);
    }
}
