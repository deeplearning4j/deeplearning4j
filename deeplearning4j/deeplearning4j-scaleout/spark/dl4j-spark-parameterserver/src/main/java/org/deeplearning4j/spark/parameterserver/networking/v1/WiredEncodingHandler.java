/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

package org.deeplearning4j.spark.parameterserver.networking.v1;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.optimize.solvers.accumulation.EncodingHandler;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.ResidualPostProcessor;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.ThresholdAlgorithm;
import org.deeplearning4j.spark.parameterserver.networking.v1.messages.SilentUpdatesMessage;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.VoidParameterServer;

import java.util.concurrent.atomic.AtomicLong;

/**
 * This MessageHandler implementation does the same as EncodingHandler, plus additionally:
 * sends encoded messages over the wire + receives encoded messages from outer parties
 *
 * @author raver119@gmail.com
 */
@Slf4j
@Deprecated
public class WiredEncodingHandler extends EncodingHandler {
    protected AtomicLong updatesCounter = new AtomicLong(0);

    /**
     * This method builds new WiredEncodingHandler instance
     *
     * @param thresholdAlgorithm threshold algorithm to use
     * @param boundary
     */
    public WiredEncodingHandler(ThresholdAlgorithm thresholdAlgorithm, ResidualPostProcessor residualPostProcessor, Integer boundary, boolean encodingDebugMode) {
        super(thresholdAlgorithm, residualPostProcessor, boundary, encodingDebugMode);
    }

    /**
     * This method sends given message to all registered recipients
     *
     * @param message
     */
    @Override
    protected void sendMessage(INDArray message, int iterationNumber, int epochNumber) {
        // here we'll send our stuff to other executores over the wire
        // and let's pray for udp broadcast availability

        // Send this message away
        // FIXME: do something with unsafe duplication, which is bad and used ONLY for local spark
        try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            long updateId = updatesCounter.getAndIncrement();

            VoidParameterServer.getInstance().execDistributedImmediately(
                            new SilentUpdatesMessage(message.unsafeDuplication(), updateId));
        }


        // heere we update local queue
        super.sendMessage(message, iterationNumber, epochNumber);
    }
}
