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

package org.nd4j.parameterserver.distributed.v2.messages.impl.base;

import lombok.Getter;
import lombok.Setter;
import org.agrona.concurrent.UnsafeBuffer;
import org.nd4j.linalg.util.SerializationUtils;
import org.nd4j.parameterserver.distributed.v2.messages.VoidMessage;

public abstract class BaseVoidMessage implements VoidMessage {
    /**
     * Unique messageId used to distringuish chunks from each other
     */
    @Getter
    @Setter
    protected String messageId;

    @Getter
    @Setter
    protected String originatorId;
}
