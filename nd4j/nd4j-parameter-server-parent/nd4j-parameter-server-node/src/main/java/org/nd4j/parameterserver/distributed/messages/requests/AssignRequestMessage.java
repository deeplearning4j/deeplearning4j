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
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.RequestMessage;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedAssignMessage;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class AssignRequestMessage extends BaseVoidMessage implements RequestMessage {

    protected Integer key;

    protected int rowIdx;

    // assign part
    protected INDArray payload;
    protected Number value;


    protected AssignRequestMessage() {
        super(8);
    }


    public AssignRequestMessage(@NonNull Integer key, @NonNull INDArray array) {
        this();
        this.key = key;
        this.payload = array.isView() ? array.dup(array.ordering()) : array;
    }

    public AssignRequestMessage(@NonNull Integer key, @NonNull Number value, int rowIdx) {
        this();
        this.key = key;
        this.value = value;
        this.rowIdx = rowIdx;
    }

    @Override
    public void processMessage() {
        if (payload == null) {
            DistributedAssignMessage dam = new DistributedAssignMessage(key, rowIdx, value.doubleValue());
            dam.extractContext(this);
            dam.processMessage();
            transport.sendMessageToAllShards(dam);
        } else {
            DistributedAssignMessage dam = new DistributedAssignMessage(key, payload);
            dam.extractContext(this);
            dam.processMessage();
            transport.sendMessageToAllShards(dam);
        }
    }
}
