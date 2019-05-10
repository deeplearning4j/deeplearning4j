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
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.DistributedMessage;
import org.nd4j.parameterserver.distributed.messages.aggregations.VectorAggregation;

/**
 * @author raver119@gmail.com
 */
@Data
@Slf4j
@Deprecated
public class DistributedVectorMessage extends BaseVoidMessage implements DistributedMessage {
    protected int rowIndex;
    protected int key;

    public DistributedVectorMessage() {
        messageType = 20;
    }

    public DistributedVectorMessage(@NonNull Integer key, int rowIndex) {
        this();
        this.rowIndex = rowIndex;
        this.key = key;
    }

    /**
     * This method will be started in context of executor, either Shard, Client or Backup node
     */
    @Override
    public void processMessage() {
        VectorAggregation aggregation = new VectorAggregation(rowIndex, (short) voidConfiguration.getNumberOfShards(),
                        shardIndex, storage.getArray(key).getRow(rowIndex).dup());
        aggregation.setOriginatorId(this.getOriginatorId());
        transport.sendMessage(aggregation);
    }
}
