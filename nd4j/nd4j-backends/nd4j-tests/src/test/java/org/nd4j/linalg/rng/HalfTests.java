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

package org.nd4j.linalg.rng;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import static junit.framework.TestCase.assertTrue;

@Slf4j
@RunWith(Parameterized.class)
public class HalfTests extends BaseNd4jTest {

    private DataType initialType;

    public HalfTests(Nd4jBackend backend) {
        super(backend);
    }

    @BeforeEach
    public void setUp() {
        if (!Nd4j.getExecutioner().getClass().getSimpleName().toLowerCase().contains("cuda"))
            return;

        initialType = Nd4j.dataType();
        Nd4j.setDataType(DataType.HALF);
    }

    @AfterEach
    public void tearDown() {
        if (!Nd4j.getExecutioner().getClass().getSimpleName().toLowerCase().contains("cuda"))
            return;

        Nd4j.setDataType(initialType);
    }

    @Test
    public void testRandomNorman_1() {
        val array = Nd4j.randn(new long[]{20, 30});

        val sum = Transforms.abs(array).sumNumber().doubleValue();

        assertTrue(sum > 0.0);
    }

    public char ordering() {
        return 'c';
    }

}
