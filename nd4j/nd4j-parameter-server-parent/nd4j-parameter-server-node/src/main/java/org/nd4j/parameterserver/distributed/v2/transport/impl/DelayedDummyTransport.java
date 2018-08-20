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

package org.nd4j.parameterserver.distributed.v2.transport.impl;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.RandomUtils;
import org.apache.commons.lang3.SerializationUtils;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.v2.messages.VoidMessage;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;

@Slf4j
public class DelayedDummyTransport extends DummyTransport {

    public DelayedDummyTransport(@NonNull String id, @NonNull Connector connector) {
        super(id, connector);
    }

    public DelayedDummyTransport(@NonNull String id, @NonNull Connector connector, @NonNull String rootId) {
        super(id, connector, rootId);
    }

    public DelayedDummyTransport(@NonNull String id, @NonNull Connector connector, @NonNull String rootId, @NonNull VoidConfiguration configuration) {
        super(id, connector, rootId, configuration);
    }

    @Override
    public void sendMessage(@NonNull VoidMessage message, @NonNull String id) {
        if (message.getOriginatorId() == null)
            message.setOriginatorId(this.id());

        val bos = new ByteArrayOutputStream();
        SerializationUtils.serialize(message, bos);

        val bis = new ByteArrayInputStream(bos.toByteArray());
        final VoidMessage msg = SerializationUtils.deserialize(bis);

        //super.sendMessage(message, id);
        connector.executorService().submit(new Runnable() {
            @Override
            public void run() {
                try {
                    // imitate some bad network here
                    val sleepTime = RandomUtils.nextInt(1, 10);
                    Thread.sleep(sleepTime);

                    DelayedDummyTransport.super.sendMessage(msg, id);
                } catch (Exception e) {
                    //
                }
            }
        });
    }
}
