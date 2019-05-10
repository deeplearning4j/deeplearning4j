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
import org.nd4j.parameterserver.distributed.messages.requests.CbowRequestMessage;

/**
 * @author raver119@gmail.com
 */
@Data
@Slf4j
public class CbowChain implements Chain {
    protected long originatorId;
    protected long taskId;
    protected long frameId;

    protected CbowRequestMessage cbowRequest;
    protected DotAggregation dotAggregation;

    public CbowChain(@NonNull CbowRequestMessage message) {
        this(message.getTaskId(), message);
    }

    public CbowChain(long taskId, @NonNull CbowRequestMessage message) {
        this.taskId = taskId;
        this.originatorId = message.getOriginatorId();
        this.frameId = message.getFrameId();
    }

    @Override
    public void addElement(VoidMessage message) {
        if (message instanceof CbowRequestMessage) {

            cbowRequest = (CbowRequestMessage) message;
        } else if (message instanceof DotAggregation) {

            dotAggregation = (DotAggregation) message;
        } else
            throw new ND4JIllegalStateException("Unknown message passed: " + message.getClass().getCanonicalName());
    }
}
