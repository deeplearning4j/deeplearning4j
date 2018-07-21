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

package org.nd4j.parameterserver.distributed.messages.aggregations;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.messages.VoidAggregation;
import org.nd4j.parameterserver.distributed.messages.complete.InitializationCompleteMessage;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class InitializationAggregation extends BaseAggregation {

    protected InitializationAggregation() {
        super();
    }

    public InitializationAggregation(int aggregationWidth, int shardIndex) {
        this((short) aggregationWidth, (short) shardIndex);
    }

    public InitializationAggregation(short aggregationWidth, short shardIndex) {
        super(-119L, aggregationWidth, shardIndex);
        this.payload = Nd4j.scalar(1.0);
    }

    @Override
    public void processMessage() {
        //log.info("sI_{} received init aggregation", transport.getShardIndex());
        if (clipboard.isTracking(this.originatorId, taskId)) {
            clipboard.pin(this);

            if (clipboard.isReady(this.originatorId, taskId)) {
                InitializationAggregation aggregation =
                                (InitializationAggregation) clipboard.unpin(this.originatorId, taskId);

                InitializationCompleteMessage icm = new InitializationCompleteMessage(taskId);
                icm.setOriginatorId(aggregation.getOriginatorId());
                transport.sendMessage(icm);
            }
        }
    }
}
