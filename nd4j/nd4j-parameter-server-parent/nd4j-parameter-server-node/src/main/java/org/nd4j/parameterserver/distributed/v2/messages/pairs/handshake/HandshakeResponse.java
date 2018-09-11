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

package org.nd4j.parameterserver.distributed.v2.messages.pairs.handshake;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.distributed.v2.messages.impl.base.BaseResponseMessage;
import org.nd4j.parameterserver.distributed.v2.util.MeshOrganizer;

@Slf4j
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class HandshakeResponse extends BaseResponseMessage {

    private long sequenceId;

    @Getter
    @Setter
    private MeshOrganizer mesh;

    /**
     * This method returns true if our node failed earlier, and should re-acquire model/updater/whatever params
     */
    @Getter
    @Setter
    @Builder.Default
    private boolean restart = false;

    /**
     * This method returns true if our node failed too many times, so it'll just enable bypass for the rest of data
     */
    @Getter
    @Setter
    private boolean dead = false;
}
