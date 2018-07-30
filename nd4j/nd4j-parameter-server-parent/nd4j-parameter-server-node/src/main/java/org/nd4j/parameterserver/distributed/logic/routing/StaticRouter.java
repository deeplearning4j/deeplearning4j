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
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * Static router implementation, the same Shard will be used for all messages
 *
 * PLEASE NOTE: Never use this router in real world! It's suitable for debugging only.
 *
 * @author raver119@gmail.com
 */
public class StaticRouter extends BaseRouter {
    protected short targetIndex;

    public StaticRouter(int targetIndex) {
        this.targetIndex = (short) targetIndex;
    }

    public StaticRouter(short targetIndex) {
        this.targetIndex = targetIndex;
    }

    @Override
    public void init(@NonNull VoidConfiguration voidConfiguration, @NonNull Transport transport) {
        super.init(voidConfiguration, transport);
    }

    @Override
    public int assignTarget(TrainingMessage message) {
        setOriginator(message);
        message.setTargetId(targetIndex);
        return targetIndex;
    }

    @Override
    public int assignTarget(VoidMessage message) {
        setOriginator(message);
        message.setTargetId(targetIndex);
        return targetIndex;
    }
}
