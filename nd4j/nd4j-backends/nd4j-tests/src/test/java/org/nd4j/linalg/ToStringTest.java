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

package org.nd4j.linalg;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

@RunWith(Parameterized.class)
@Slf4j
public class ToStringTest extends BaseNd4jTest {
    public ToStringTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testToString() throws Exception {
        assertEquals("[         1,         2,         3]",
                Nd4j.createFromArray(1, 2, 3).toString());

        assertEquals("[       1,       2,       3,       4,       5,       6,       7,       8]",
                Nd4j.createFromArray(1, 2, 3, 4, 5, 6, 7, 8).toString(1000, false, 2));

        assertEquals("[    1.132,    2.644,    3.234]",
                Nd4j.createFromArray(1.132414, 2.64356456, 3.234234).toString(1000, false, 3));

        assertEquals("[               1.132414,             2.64356456,             3.25345234]",
                Nd4j.createFromArray(1.132414, 2.64356456, 3.25345234).toStringFull());

        assertEquals("[      1,      2,      3,  ...      6,      7,      8]",
                Nd4j.createFromArray(1, 2, 3, 4, 5, 6, 7, 8).toString(6, true, 1));
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
