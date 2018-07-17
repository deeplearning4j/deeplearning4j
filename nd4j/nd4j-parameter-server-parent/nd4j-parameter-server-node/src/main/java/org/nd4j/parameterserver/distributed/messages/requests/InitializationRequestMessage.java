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

package org.nd4j.parameterserver.distributed.messages.requests;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.RequestMessage;
import org.nd4j.parameterserver.distributed.messages.aggregations.InitializationAggregation;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedInitializationMessage;

/**
 * This method propagates storage/weights initialization over distributed Shards
 *
 * @author raver119@gmail.com
 */
@Slf4j
@Builder
public class InitializationRequestMessage extends BaseVoidMessage implements RequestMessage {
    protected int vectorLength;
    protected int numWords;
    protected long seed;
    protected boolean useHs;
    protected boolean useNeg;
    protected int columnsPerShard;

    protected InitializationRequestMessage() {
        super(4);
        taskId = -119L;
    }

    public InitializationRequestMessage(int vectorLength, int numWords, long seed, boolean useHs, boolean useNeg,
                    int columnsPerShard) {
        this();
        this.vectorLength = vectorLength;
        this.numWords = numWords;
        this.seed = seed;
        this.useHs = useHs;
        this.useNeg = useNeg;
        this.columnsPerShard = columnsPerShard;
    }


    @Override
    public void processMessage() {
        DistributedInitializationMessage dim = new DistributedInitializationMessage(vectorLength, numWords, seed, useHs,
                        useNeg, columnsPerShard);

        InitializationAggregation aggregation = new InitializationAggregation(
                        (short) voidConfiguration.getNumberOfShards(), transport.getShardIndex());
        aggregation.setOriginatorId(this.originatorId);

        clipboard.pin(aggregation);

        dim.setOriginatorId(this.originatorId);
        dim.extractContext(this);
        dim.processMessage();

        if (voidConfiguration.getNumberOfShards() > 1)
            transport.sendMessageToAllShards(dim);
    }

    @Override
    public boolean isBlockingMessage() {
        return true;
    }
}
