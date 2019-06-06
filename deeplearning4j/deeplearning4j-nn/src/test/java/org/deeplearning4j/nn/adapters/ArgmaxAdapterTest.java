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

package org.deeplearning4j.nn.adapters;

import lombok.val;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class ArgmaxAdapterTest {
    @Test
    public void testSoftmax_2D_1() {
        val in = new double[][] {{1, 3, 2}, { 4, 5, 6}};

        val adapter = new ArgmaxAdapter();
        val result = adapter.apply(Nd4j.create(in));

        assertArrayEquals(new int[]{1, 2}, result);
    }

    @Test
    public void testSoftmax_1D_1() {
        val in = new double[] {1, 3, 2};

        val adapter = new ArgmaxAdapter();
        val result = adapter.apply(Nd4j.create(in));

        assertArrayEquals(new int[]{1}, result);
    }
}