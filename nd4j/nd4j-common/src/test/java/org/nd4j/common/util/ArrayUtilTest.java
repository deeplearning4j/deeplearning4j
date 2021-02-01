/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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

import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;
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

}
