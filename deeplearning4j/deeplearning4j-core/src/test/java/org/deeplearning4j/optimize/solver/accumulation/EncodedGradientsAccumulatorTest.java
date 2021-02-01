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

package org.deeplearning4j.optimize.solver.accumulation;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.optimize.solvers.accumulation.EncodedGradientsAccumulator;
import org.deeplearning4j.optimize.solvers.accumulation.EncodingHandler;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold.FixedThresholdAlgorithm;
import org.junit.Test;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.util.PrintAffinity;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

@Slf4j
public class EncodedGradientsAccumulatorTest extends BaseDL4JTest {

    @Override
    public long getTimeoutMilliseconds() {
        return 1200000L;
    }

    /**
     * This test ensures, that memory amount assigned to buffer is enough for any number of updates
     * @throws Exception
     */
    @Test
    public void testStore1() throws Exception {
        int numParams;
        int[] workers;
        if(isIntegrationTests()){
            numParams = 100000;
            workers = new int[] {2, 4, 8};
        } else {
            numParams = 10000;
            workers = new int[] {2, 3};
        }

        for (int numWorkers : workers) {
            EncodingHandler handler = new EncodingHandler(new FixedThresholdAlgorithm(1e-3),null, null, false);

            val bufferSize = EncodedGradientsAccumulator.getOptimalBufferSize(numParams, numWorkers, 2);
            log.info("Workers: {}; Buffer size: {} bytes", numWorkers, bufferSize);
            EncodedGradientsAccumulator accumulator =
                            new EncodedGradientsAccumulator(numWorkers, handler, bufferSize, 2, null, false);

            for (int e = 10; e < numParams / 10; e++) {
                INDArray encoded = handler.encodeUpdates(0, 0, getGradients(numParams, e, 2e-3));
                accumulator.receiveUpdate(encoded);

                // just purge updates, like they were consumed
                for (int i = 0; i < accumulator.getMessages().size(); i++) {
                    accumulator.getMessages().get(i).clear();
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
        int numParams;
        if(isIntegrationTests()){
            numParams = 100000;
        } else {
            numParams = 10000;
        }


        EncodingHandler handler = new EncodingHandler(new FixedThresholdAlgorithm(1e-3), null, Integer.MAX_VALUE, false);
        for (int e = 10; e < numParams / 5; e++) {

            val gradients = getGradients(numParams, e, 2e-3);
            val encoded = handler.encodeUpdates(0, 0, gradients);

            assertNotNull("Failed with e == " + e, encoded);

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
