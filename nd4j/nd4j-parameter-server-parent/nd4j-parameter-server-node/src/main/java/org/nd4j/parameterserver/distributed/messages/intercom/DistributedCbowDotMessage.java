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

package org.nd4j.parameterserver.distributed.messages.intercom;

import lombok.Data;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.enums.ExecutionMode;
import org.nd4j.parameterserver.distributed.logic.storage.WordVectorStorage;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.DistributedMessage;
import org.nd4j.parameterserver.distributed.messages.aggregations.DotAggregation;
import org.nd4j.parameterserver.distributed.messages.requests.CbowRequestMessage;
import org.nd4j.parameterserver.distributed.training.impl.CbowTrainer;

import java.util.Arrays;

/**
 * @author raver119@gmail.com
 */
@Data
@Slf4j
public class DistributedCbowDotMessage extends BaseVoidMessage implements DistributedMessage {
    protected int[] rowsA;
    protected int[] rowsB;

    // payload for trainer pickup
    protected int w1;
    protected boolean useHS;
    protected short negSamples;
    protected float alpha;
    protected byte[] codes;

    public DistributedCbowDotMessage() {
        messageType = 22;
    }

    @Deprecated
    public DistributedCbowDotMessage(long taskId, int rowA, int rowB) {
        this(taskId, new int[] {rowA}, new int[] {rowB}, rowA, new byte[] {}, false, (short) 0, 0.001f);
    }

    public DistributedCbowDotMessage(long taskId, @NonNull int[] rowsA, @NonNull int[] rowsB, int w1,
                    @NonNull byte[] codes, boolean useHS, short negSamples, float alpha) {
        this();
        this.rowsA = rowsA;
        this.rowsB = rowsB;
        this.taskId = taskId;

        this.w1 = w1;
        this.useHS = useHS;
        this.negSamples = negSamples;
        this.alpha = alpha;
        this.codes = codes;


        //if (this.rowsA.length != this.rowsB.length)
        //    throw new ND4JIllegalStateException("Length of X should match length of Y");
    }

    /**
     * This method calculates dot of gives rows, with averaging applied to rowsA, as required by CBoW
     */
    @Override
    public void processMessage() {
        // this only picks up new training round
        //log.info("sI_{} Starting CBOW dot...", transport.getShardIndex());

        CbowRequestMessage cbrm = new CbowRequestMessage(rowsA, rowsB, w1, codes, negSamples, alpha, 119);
        if (negSamples > 0) {
            // unfortunately we have to get copy of negSamples here
            int negatives[] = Arrays.copyOfRange(rowsB, codes.length, rowsB.length);
            cbrm.setNegatives(negatives);
        }
        cbrm.setFrameId(-119L);
        cbrm.setTaskId(this.taskId);
        cbrm.setOriginatorId(this.getOriginatorId());


        // FIXME: get rid of THAT
        CbowTrainer cbt = (CbowTrainer) trainer;
        cbt.pickTraining(cbrm);


        // we calculate dot for all involved rows, and first of all we get mean word
        INDArray words = Nd4j.pullRows(storage.getArray(WordVectorStorage.SYN_0), 1, rowsA, 'c');
        INDArray mean = words.mean(0);

        int resultLength = codes.length + (negSamples > 0 ? (negSamples + 1) : 0);

        INDArray result = Nd4j.createUninitialized(resultLength, 1);
        int e = 0;
        for (; e < codes.length; e++) {
            double dot = Nd4j.getBlasWrapper().dot(mean, storage.getArray(WordVectorStorage.SYN_1).getRow(rowsB[e]));
            result.putScalar(e, dot);
        }

        // negSampling round
        for (; e < resultLength; e++) {
            double dot = Nd4j.getBlasWrapper().dot(mean,
                            storage.getArray(WordVectorStorage.SYN_1_NEGATIVE).getRow(rowsB[e]));
            result.putScalar(e, dot);
        }

        if (voidConfiguration.getExecutionMode() == ExecutionMode.AVERAGING) {
            DotAggregation dot = new DotAggregation(taskId, (short) 1, shardIndex, result);
            dot.setTargetId((short) -1);
            dot.setOriginatorId(getOriginatorId());
            transport.putMessage(dot);
        } else if (voidConfiguration.getExecutionMode() == ExecutionMode.SHARDED) {
            // send this message to everyone
            DotAggregation dot = new DotAggregation(taskId, (short) voidConfiguration.getNumberOfShards(), shardIndex,
                            result);
            dot.setTargetId((short) -1);
            dot.setOriginatorId(getOriginatorId());
            transport.sendMessage(dot);
        }
    }
}
