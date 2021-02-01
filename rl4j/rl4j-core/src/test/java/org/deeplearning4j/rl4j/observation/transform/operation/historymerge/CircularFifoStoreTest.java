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

package org.deeplearning4j.rl4j.observation.transform.operation.historymerge;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class CircularFifoStoreTest {

    @Test(expected = IllegalArgumentException.class)
    public void when_fifoSizeIsLessThan1_expect_exception() {
        // Arrange
        CircularFifoStore sut = new CircularFifoStore(0);
    }

    @Test
    public void when_adding2elementsWithSize2_expect_notReadyAfter1stReadyAfter2nd() {
        // Arrange
        CircularFifoStore sut = new CircularFifoStore(2);
        INDArray firstElement = Nd4j.create(new double[] { 1.0, 2.0, 3.0 });
        INDArray secondElement = Nd4j.create(new double[] { 10.0, 20.0, 30.0 });

        // Act
        sut.add(firstElement);
        boolean isReadyAfter1st = sut.isReady();
        sut.add(secondElement);
        boolean isReadyAfter2nd = sut.isReady();

        // Assert
        assertFalse(isReadyAfter1st);
        assertTrue(isReadyAfter2nd);
    }

    @Test
    public void when_adding2elementsWithSize2_expect_getReturnThese2() {
        // Arrange
        CircularFifoStore sut = new CircularFifoStore(2);
        INDArray firstElement = Nd4j.create(new double[] { 1.0, 2.0, 3.0 });
        INDArray secondElement = Nd4j.create(new double[] { 10.0, 20.0, 30.0 });

        // Act
        sut.add(firstElement);
        sut.add(secondElement);
        INDArray[] results = sut.get();

        // Assert
        assertEquals(2, results.length);

        assertEquals(1.0, results[0].getDouble(0), 0.00001);
        assertEquals(2.0, results[0].getDouble(1), 0.00001);
        assertEquals(3.0, results[0].getDouble(2), 0.00001);

        assertEquals(10.0, results[1].getDouble(0), 0.00001);
        assertEquals(20.0, results[1].getDouble(1), 0.00001);
        assertEquals(30.0, results[1].getDouble(2), 0.00001);

    }

    @Test
    public void when_adding2elementsThenCallingReset_expect_getReturnEmpty() {
        // Arrange
        CircularFifoStore sut = new CircularFifoStore(2);
        INDArray firstElement = Nd4j.create(new double[] { 1.0, 2.0, 3.0 });
        INDArray secondElement = Nd4j.create(new double[] { 10.0, 20.0, 30.0 });

        // Act
        sut.add(firstElement);
        sut.add(secondElement);
        sut.reset();
        INDArray[] results = sut.get();

        // Assert
        assertEquals(0, results.length);
    }

}
