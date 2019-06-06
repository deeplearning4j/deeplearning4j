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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@Deprecated
public class DotAggregation extends BaseAggregation {

    protected DotAggregation() {
        super();
    }

    public DotAggregation(long taskId, short aggregationWidth, short shardIndex, INDArray scalar) {
        super(taskId, aggregationWidth, shardIndex);

        this.payload = scalar;
        addToChunks(payload);
    }

    @Override
    public INDArray getAccumulatedResult() {
        INDArray stack = super.getAccumulatedResult();

        if (aggregationWidth == 1)
            return stack;

        if (stack.isRowVector()) {
            return Nd4j.scalar(stack.sumNumber().doubleValue());
        } else {
            return stack.sum(1);
        }
    }

    /**
     * This method will be started in context of executor, either Shard, Client or Backup node
     */
    @Override
    public void processMessage() {
        // since our computations are symmetric - we aggregate dot everywhere
        if (chunks == null) {
            chunks = new TreeMap<>();
            chunksCounter = new AtomicInteger(1);
            addToChunks(payload);
        }

        clipboard.pin(this);

        //log.info("sI_{} dot aggregation received", transport.getShardIndex());

        if (clipboard.isReady(this.getOriginatorId(), this.getTaskId())) {
            trainer.aggregationFinished(clipboard.unpin(this.getOriginatorId(), this.taskId));
        }
    }
}
