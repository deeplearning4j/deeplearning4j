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

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.RequestMessage;
import org.nd4j.parameterserver.distributed.messages.complete.IntroductionCompleteMessage;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedIntroductionMessage;

/**
 * This message will be sent by each shard, during meeting
 *
 * @author raver119@gmail.com
 */
@Deprecated
public class IntroductionRequestMessage extends BaseVoidMessage implements RequestMessage {
    private String ip;
    private int port;

    public IntroductionRequestMessage() {
        super(5);
    }

    public IntroductionRequestMessage(@NonNull String ip, int port) {
        this();
        this.ip = ip;
        this.port = port;
    }

    @Override
    public void processMessage() {
        // redistribute this message over network
        transport.addClient(ip, port);

        //        DistributedIntroductionMessage dim = new DistributedIntroductionMessage(ip, port);

        //        dim.extractContext(this);
        //        dim.processMessage();

        //        if (voidConfiguration.getNumberOfShards() > 1)
        //            transport.sendMessageToAllShards(dim);

        //        IntroductionCompleteMessage icm = new IntroductionCompleteMessage(this.taskId);
        //        icm.setOriginatorId(this.originatorId);

        //        transport.sendMessage(icm);
    }

    @Override
    public boolean isBlockingMessage() {
        return true;
    }
}
