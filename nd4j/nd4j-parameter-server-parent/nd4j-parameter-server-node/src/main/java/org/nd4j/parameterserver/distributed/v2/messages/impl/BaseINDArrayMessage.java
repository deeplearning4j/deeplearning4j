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

package org.nd4j.parameterserver.distributed.v2.messages.impl;

import lombok.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.parameterserver.distributed.v2.messages.INDArrayMessage;

/**
 * This message holds some INDArray
 * @author raver119@gmail.com
 */
@NoArgsConstructor
@AllArgsConstructor
public abstract class BaseINDArrayMessage implements INDArrayMessage {
    private static final long serialVersionUID = 1L;

    @Getter
    protected String messageId;

    @Getter
    @Setter
    protected String originatorId;

    @Getter
    @Setter
    protected String requestId;

    @Getter
    protected INDArray payload;

    protected BaseINDArrayMessage(@NonNull String messageId, @NonNull INDArray payload) {
        this.messageId = messageId;
        this.payload = payload;
    }
}
