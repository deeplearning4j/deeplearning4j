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

package org.deeplearning4j.spark.parameterserver.networking.v1.messages;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.DistributedMessage;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class SilentIntroductoryMessage extends BaseVoidMessage implements DistributedMessage {
    protected String localIp;
    protected int port;

    protected SilentIntroductoryMessage() {
        //
    }

    public SilentIntroductoryMessage(@NonNull String localIP, int port) {
        this.localIp = localIP;
        this.port = port;
    }

    @Override
    public void processMessage() {
        /*
            basically we just want to send our IP, and get our new shardIndex in return. haha. bad idea obviously, but still...
        
            or, we can skip direct addressing here, use passive addressing instead, like in client mode?
         */

        log.info("Adding client {}:{}", localIp, port);
        //transport.addShard(localIp, port);
        transport.addClient(localIp, port);
    }

    @Override
    public boolean isBlockingMessage() {
        // this is blocking message, we want to get reply back before going further
        return true;
    }
}
