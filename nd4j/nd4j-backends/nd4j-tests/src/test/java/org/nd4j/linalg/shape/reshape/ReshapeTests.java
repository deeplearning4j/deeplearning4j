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

package org.nd4j.linalg.shape.reshape;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeNotNull;

/**
 * @author Adam Gibson
 */
@Slf4j
@RunWith(Parameterized.class)
public class ReshapeTests extends BaseNd4jTest {

    public ReshapeTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testThreeTwoTwoTwo() {
        INDArray threeTwoTwo = Nd4j.linspace(1, 12, 12).reshape(3, 2, 2);
        INDArray sliceZero = Nd4j.create(new double[][] {{1, 7}, {4, 10}});
        INDArray sliceOne = Nd4j.create(new double[][] {{2, 8}, {5, 11}});
        INDArray sliceTwo = Nd4j.create(new double[][] {{3, 9}, {6, 12}});
        INDArray[] assertions = new INDArray[] {sliceZero, sliceOne, sliceTwo};

        for (int i = 0; i < threeTwoTwo.slices(); i++) {
            INDArray sliceI = threeTwoTwo.slice(i);
            assertEquals(assertions[i], sliceI);
        }

        INDArray linspaced = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray[] assertionsTwo = new INDArray[] {Nd4j.create(new double[] {1, 3}), Nd4j.create(new double[] {2, 4})};

        for (int i = 0; i < assertionsTwo.length; i++)
            assertEquals(linspaced.slice(i), assertionsTwo[i]);
    }


    @Test
    public void testColumnVectorReshape() {
        double delta = 1e-1;
        INDArray arr = Nd4j.create(1, 3);
        INDArray reshaped = arr.reshape('f', 3, 1);
        assertArrayEquals(new int[] {3, 1}, reshaped.shape());
        assertEquals(0.0, reshaped.getDouble(1), delta);
        assertEquals(0.0, reshaped.getDouble(2), delta);
        log.info("Reshaped: {}", reshaped.shapeInfoDataBuffer().asInt());
        assumeNotNull(reshaped.toString());
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
