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

package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertTrue;

/**
 * Tests for memory-related stuff in gradients accumulator
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class EncodedGradientsAccumulatorTest {

    /**
     * This test ensures, that memory amount assigned to buffer is enough for any number of updates
     * @throws Exception
     */
    @Test
    public void testStore1() throws Exception {
        int numParams = 100000;

        int workers[] = new int[] {2, 4, 8};

        for (int numWorkers : workers) {
            EncodingHandler handler = new EncodingHandler(1e-3);

            val bufferSize = EncodedGradientsAccumulator.getOptimalBufferSize(numParams, numWorkers, 2);
            log.info("Workers: {}; Buffer size: {} bytes", numWorkers, bufferSize);
            EncodedGradientsAccumulator accumulator =
                            new EncodedGradientsAccumulator(numWorkers, handler, bufferSize, 2, null);

            for (int e = 10; e < numParams / 10; e++) {
                INDArray encoded = handler.encodeUpdates(getGradients(numParams, e, 2e-3));
                accumulator.receiveUpdate(encoded);

                // just purge updates, like they were consumed
                for (int i = 0; i < accumulator.messages.size(); i++) {
                    accumulator.messages.get(i).clear();
                }
            }
        }
    }


    /**
     * Here we ensure that no matter how dense/sparse our updates are - we're never going above 1/16 of original elements of gradients array
     *
     * @throws Exception
     */
    @Test
    public void testEncodingLimits1() throws Exception {
        int numParams = 100000;

        for (int e = 10; e < numParams / 5; e++) {
            EncodingHandler handler = new EncodingHandler(1e-3);

            INDArray encoded = handler.encodeUpdates(getGradients(numParams, e, 2e-3));

            //  log.info("enc len: {}", encoded.data().length());

            int encFormat = encoded.data().getInt(3);

            assertTrue("Failed for E = " + e + "; Format: " + encFormat + "; Length: " + encoded.data().length(),
                            encoded.data().length() < numParams / 16 + 6);
        }
    }


    protected INDArray getGradients(int length, int numPositives, double value) {
        INDArray grad = Nd4j.create(length);

        for (int i = 0; i < numPositives; i++) {
            grad.putScalar(i, value);
        }

        return grad;
    }
}
