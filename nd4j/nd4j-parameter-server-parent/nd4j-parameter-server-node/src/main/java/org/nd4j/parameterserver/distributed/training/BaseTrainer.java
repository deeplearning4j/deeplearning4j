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

package org.nd4j.parameterserver.distributed.training;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.logic.completion.Clipboard;
import org.nd4j.parameterserver.distributed.logic.Storage;
import org.nd4j.parameterserver.distributed.logic.completion.FrameCompletionHandler;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * @author raver119@gmail.co,
 */
@Deprecated
public abstract class BaseTrainer<T extends TrainingMessage> implements TrainingDriver<T> {
    protected VoidConfiguration voidConfiguration;
    protected Transport transport;
    protected Clipboard clipboard;
    protected Storage storage;

    protected FrameCompletionHandler completionHandler = new FrameCompletionHandler();

    @Override
    public void init(@NonNull VoidConfiguration voidConfiguration, @NonNull Transport transport,
                    @NonNull Storage storage, @NonNull Clipboard clipboard) {
        this.clipboard = clipboard;
        this.transport = transport;
        this.voidConfiguration = voidConfiguration;
        this.storage = storage;
    }

    protected int[] replicate(int value, int size) {
        int[] result = new int[size];
        for (int e = 0; e < size; e++)
            result[e] = value;

        return result;
    }

    @Override
    public void addCompletionHook(long originatorId, long frameId, long messageId) {
        completionHandler.addHook(originatorId, frameId, messageId);
    }
}
