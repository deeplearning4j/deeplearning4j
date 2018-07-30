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

package org.nd4j.parameterserver.distributed.logic.routing;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.RandomUtils;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.messages.Frame;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.messages.requests.SkipGramRequestMessage;
import org.nd4j.parameterserver.distributed.transport.Transport;

import java.util.concurrent.atomic.AtomicLong;

/**
 * This is main router implementation for VoidParameterServer
 * Basic idea: We route TrainingMessages conditionally, based on Huffman tree index (aka frequency-ordered position)
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class InterleavedRouter extends BaseRouter {
    protected short targetIndex = (short) -1;
    protected AtomicLong counter = new AtomicLong(0);

    public InterleavedRouter() {

    }

    public InterleavedRouter(int defaultIndex) {
        this();
        this.targetIndex = (short) defaultIndex;
    }

    @Override
    public void init(@NonNull VoidConfiguration voidConfiguration, @NonNull Transport transport) {
        super.init(voidConfiguration, transport);

        // by default messages are being routed to any random shard
        if (targetIndex < 0)
            targetIndex = (short) RandomUtils.nextInt(0, voidConfiguration.getNumberOfShards());
    }

    @Override
    public int assignTarget(TrainingMessage message) {
        setOriginator(message);
        if (message instanceof SkipGramRequestMessage) {
            SkipGramRequestMessage sgrm = (SkipGramRequestMessage) message;

            int w1 = sgrm.getW1();
            if (w1 >= voidConfiguration.getNumberOfShards())
                message.setTargetId((short) (w1 % voidConfiguration.getNumberOfShards()));
            else
                message.setTargetId((short) w1);
        } else {
            message.setTargetId((short) Math.abs(counter.incrementAndGet() % voidConfiguration.getNumberOfShards()));
        }

        return message.getTargetId();
    }

    @Override
    public int assignTarget(VoidMessage message) {
        setOriginator(message);
        if (message instanceof Frame) {
            message.setTargetId((short) Math.abs(counter.incrementAndGet() % voidConfiguration.getNumberOfShards()));
        } else
            message.setTargetId(targetIndex);
        return message.getTargetId();
    }
}
