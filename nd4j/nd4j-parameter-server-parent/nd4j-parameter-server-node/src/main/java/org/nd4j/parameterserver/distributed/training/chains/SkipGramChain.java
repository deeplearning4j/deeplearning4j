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

package org.nd4j.parameterserver.distributed.training.chains;

import lombok.Data;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.parameterserver.distributed.messages.Chain;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.messages.aggregations.DotAggregation;
import org.nd4j.parameterserver.distributed.messages.requests.SkipGramRequestMessage;

/**
 * Chain implementation for SkipGram
 *
 * @author raver119@gmail.com
 */
@Data
@Slf4j
public class SkipGramChain implements Chain {

    protected long originatorId;
    protected long taskId;
    protected long frameId;

    protected SkipGramRequestMessage requestMessage;
    protected DotAggregation dotAggregation;

    public SkipGramChain(long originatorId, long taskId, long frameId) {
        this.taskId = taskId;
        this.frameId = frameId;
        this.originatorId = originatorId;
    }

    public SkipGramChain(@NonNull SkipGramRequestMessage message) {
        this(message.getTaskId(), message);
    }

    public SkipGramChain(long taskId, @NonNull SkipGramRequestMessage message) {
        this(message.getOriginatorId(), taskId, message.getFrameId());
        addElement(message);
    }

    @Override
    public long getTaskId() {
        return taskId;
    }

    @Override
    public void addElement(VoidMessage message) {
        if (message instanceof SkipGramRequestMessage) {
            requestMessage = (SkipGramRequestMessage) message;

        } else if (message instanceof DotAggregation) {
            dotAggregation = (DotAggregation) message;

        } else
            throw new ND4JIllegalStateException(
                            "Unknown message received: [" + message.getClass().getCanonicalName() + "]");
    }
}
