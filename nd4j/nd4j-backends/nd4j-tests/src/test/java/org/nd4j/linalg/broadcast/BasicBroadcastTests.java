/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.linalg.broadcast;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class BasicBroadcastTests extends BaseNd4jTest {
    public BasicBroadcastTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void basicBroadcastTest_1() {
        val x = Nd4j.create(DataType.FLOAT, 3, 5);
        val y = Nd4j.createFromArray(new float[]{1.f, 1.f, 1.f, 1.f, 1.f});
        val e = Nd4j.create(DataType.FLOAT, 3, 5).assign(1.f);

        // inplace setup
        val op = new AddOp(new INDArray[]{x, y}, new INDArray[]{x});

        Nd4j.exec(op);

        Nd4j.getExecutioner().commit();

        assertEquals(e, x);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
