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

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.Setter;
import org.nd4j.parameterserver.distributed.v2.messages.BroadcastableMessage;
import org.nd4j.parameterserver.distributed.v2.messages.impl.base.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.v2.util.MeshOrganizer;

/**
 * This message is used to send Mesh state to all nodes within network
 * @author raver119@gmail.com
 */
@NoArgsConstructor
public class MeshUpdateMessage extends BaseVoidMessage implements BroadcastableMessage {
    private static final long serialVersionUID = 1L;

    @Getter
    @Setter
    private String relayId;

    @Getter
    private MeshOrganizer mesh;

    public MeshUpdateMessage(@NonNull MeshOrganizer mesh) {
        this.mesh = mesh;
    }
}
