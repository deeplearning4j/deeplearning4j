/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.util;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

/**
 *
 */
public class ArrayUtilTest extends BaseDL4JTest {

    @Test
    public void testRange() {
        int[] range = ArrayUtil.range(0, 2);
        int[] test = {0, 1};
        assertEquals(true, Arrays.equals(test, range));

        int[] test2 = {-1, 0};
        int[] range2 = ArrayUtil.range(-1, 1);
        assertEquals(true, Arrays.equals(test2, range2));

    }

    @Test
    public void testStrides() {
        int[] shape = {5, 4, 3};
        int[] cStyleStride = {12, 3, 1};
        int[] fortranStyleStride = {1, 5, 20};
        int[] fortranStyleTest = ArrayUtil.calcStridesFortran(shape);
        int[] cStyleTest = ArrayUtil.calcStrides(shape);
        assertEquals(true, Arrays.equals(cStyleStride, cStyleTest));
        assertEquals(true, Arrays.equals(fortranStyleStride, fortranStyleTest));

        int[] shape2 = {2, 2};
        int[] cStyleStride2 = {2, 1};
        int[] fortranStyleStride2 = {1, 2};
        int[] cStyleTest2 = ArrayUtil.calcStrides(shape2);
        int[] fortranStyleTest2 = ArrayUtil.calcStridesFortran(shape2);
        assertEquals(true, Arrays.equals(cStyleStride2, cStyleTest2));
        assertEquals(true, Arrays.equals(fortranStyleStride2, fortranStyleTest2));



    }


}
