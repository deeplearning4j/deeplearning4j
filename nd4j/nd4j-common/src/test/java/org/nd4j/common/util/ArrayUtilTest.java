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

package org.nd4j.common.util;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;
import org.nd4j.common.util.ArrayUtil;

public class ArrayUtilTest {

    @Test
    public void testInvertPermutationInt(){
        assertArrayEquals(
                new int[]{ 2, 4, 3, 0, 1 },
                ArrayUtil.invertPermutation(3, 4, 0, 2, 1)
                );
    }

    @Test
    public void testInvertPermutationLong(){
        assertArrayEquals(
                new long[]{ 2, 4, 3, 0, 1 },
                ArrayUtil.invertPermutation(3L, 4L, 0L, 2L, 1L)
        );
    }

    @Test
    public void testCalcStridesFortranIntEmptyShape() {
        // Test that calcStridesFortran returns [1] for empty shapes (length 0)
        assertArrayEquals(
                new int[]{1},
                ArrayUtil.calcStridesFortran(new int[]{})
        );
    }

    @Test
    public void testCalcStridesFortranIntSingleElementShape() {
        // Test that calcStridesFortran returns [1] for single-element shapes (length 1)
        assertArrayEquals(
                new int[]{1},
                ArrayUtil.calcStridesFortran(new int[]{5})
        );
    }

    @Test
    public void testCalcStridesFortranLongEmptyShape() {
        // Test that calcStridesFortran returns [1] for empty shapes (length 0)
        assertArrayEquals(
                new long[]{1},
                ArrayUtil.calcStridesFortran(new long[]{})
        );
    }

    @Test
    public void testCalcStridesFortranLongSingleElementShape() {
        // Test that calcStridesFortran returns [1] for single-element shapes (length 1)
        assertArrayEquals(
                new long[]{1},
                ArrayUtil.calcStridesFortran(new long[]{5L})
        );
    }

    @Test
    public void testCalcStridesIntEmptyShape() {
        // Test that calcStrides returns [1] for empty shapes (length 0)
        assertArrayEquals(
                new int[]{1},
                ArrayUtil.calcStrides(new int[]{})
        );
    }

    @Test
    public void testCalcStridesIntSingleElementShape() {
        // Test that calcStrides returns [1] for single-element shapes (length 1)
        assertArrayEquals(
                new int[]{1},
                ArrayUtil.calcStrides(new int[]{5})
        );
    }

    @Test
    public void testCalcStridesLongEmptyShape() {
        // Test that calcStrides returns [1] for empty shapes (length 0)
        assertArrayEquals(
                new long[]{1},
                ArrayUtil.calcStrides(new long[]{})
        );
    }

    @Test
    public void testCalcStridesLongSingleElementShape() {
        // Test that calcStrides returns [1] for single-element shapes (length 1)
        assertArrayEquals(
                new long[]{1},
                ArrayUtil.calcStrides(new long[]{5L})
        );
    }

}
