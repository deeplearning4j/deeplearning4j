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

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.shape.Concat;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.*;

@Slf4j
@RunWith(Parameterized.class)
public class EmptyTests extends BaseNd4jTest {

    DataBuffer.Type initialType;

    public EmptyTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }


    @Test
    public void testEmpyArray_1() {
        val array = Nd4j.empty();

        assertNotNull(array);
        assertTrue(array.isEmpty());

        assertFalse(array.isScalar());
        assertFalse(array.isVector());
        assertFalse(array.isRowVector());
        assertFalse(array.isColumnVector());
        assertFalse(array.isCompressed());
        assertFalse(array.isSparse());

        assertFalse(array.isAttached());

        assertEquals(Nd4j.dataType(), array.dataType());
    }


    @Test
    public void testEmptyDtype_1() {
        val array = Nd4j.empty(DataBuffer.Type.INT);

        assertTrue(array.isEmpty());
        assertEquals(DataBuffer.Type.INT, array.dataType());
    }

    @Test
    public void testEmptyDtype_2() {
        val array = Nd4j.empty(DataBuffer.Type.LONG);

        assertTrue(array.isEmpty());
        assertEquals(DataBuffer.Type.LONG, array.dataType());
    }

    @Test
    public void testConcat_1() {
        val row1 = Nd4j.create(new double[]{1, 1, 1, 1}, new long[]{1, 4});
        val row2 = Nd4j.create(new double[]{2, 2, 2, 2}, new long[]{1, 4});
        val row3 = Nd4j.create(new double[]{3, 3, 3, 3}, new long[]{1, 4});

        val exp = Nd4j.create(new double[]{1, 1, 1, 1,    2, 2, 2, 2,   3, 3, 3, 3}, new int[]{3, 4});

        val op = DynamicCustomOp.builder("concat")
                .addInputs(row1, row2, row3)
                .addIntegerArguments(0)
                .build();

        Nd4j.getExecutioner().exec(op);

        val z = op.getOutputArgument(0);

        assertEquals(exp, z);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
