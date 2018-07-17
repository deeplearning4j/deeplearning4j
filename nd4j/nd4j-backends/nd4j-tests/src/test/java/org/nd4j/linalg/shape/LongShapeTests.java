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

package org.nd4j.linalg.shape;

import lombok.val;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * This class contains tests for new Long shapes
 * @author raver119@gmail.com
 */

@RunWith(Parameterized.class)
public class LongShapeTests extends BaseNd4jTest {

    public LongShapeTests(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testLongBuffer_1() {
        val exp = new long[]{2, 5, 3, 3, 1, 0, 1, 99};
        val buffer = Nd4j.getDataBufferFactory().createLong(exp);

        val java = buffer.asLong();

        assertArrayEquals(exp, java);
    }


    @Test
    public void testLongShape_1() {
        val exp = new long[]{2, 5, 3, 3, 1, 0, 1, 99};

        val array = Nd4j.createUninitialized(5, 3);
        val buffer = array.shapeInfoDataBuffer();

        val java = buffer.asLong();

        assertArrayEquals(exp, java);
        assertEquals(8, buffer.getElementSize());
    }



    @Override
    public char ordering() {
        return 'c';
    }
}
