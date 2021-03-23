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
package org.deeplearning4j.util;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.common.util.ArrayUtil;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;

/**
 */
@DisplayName("Array Util Test")
@Tag(TagNames.JAVA_ONLY)
class ArrayUtilTest extends BaseDL4JTest {

    @Test
    @DisplayName("Test Range")
    void testRange() {
        int[] range = ArrayUtil.range(0, 2);
        int[] test = { 0, 1 };
        assertEquals(true, Arrays.equals(test, range));
        int[] test2 = { -1, 0 };
        int[] range2 = ArrayUtil.range(-1, 1);
        assertEquals(true, Arrays.equals(test2, range2));
    }

    @Test
    @DisplayName("Test Strides")
    void testStrides() {
        int[] shape = { 5, 4, 3 };
        int[] cStyleStride = { 12, 3, 1 };
        int[] fortranStyleStride = { 1, 5, 20 };
        int[] fortranStyleTest = ArrayUtil.calcStridesFortran(shape);
        int[] cStyleTest = ArrayUtil.calcStrides(shape);
        assertEquals(true, Arrays.equals(cStyleStride, cStyleTest));
        assertEquals(true, Arrays.equals(fortranStyleStride, fortranStyleTest));
        int[] shape2 = { 2, 2 };
        int[] cStyleStride2 = { 2, 1 };
        int[] fortranStyleStride2 = { 1, 2 };
        int[] cStyleTest2 = ArrayUtil.calcStrides(shape2);
        int[] fortranStyleTest2 = ArrayUtil.calcStridesFortran(shape2);
        assertEquals(true, Arrays.equals(cStyleStride2, cStyleTest2));
        assertEquals(true, Arrays.equals(fortranStyleStride2, fortranStyleTest2));
    }
}
