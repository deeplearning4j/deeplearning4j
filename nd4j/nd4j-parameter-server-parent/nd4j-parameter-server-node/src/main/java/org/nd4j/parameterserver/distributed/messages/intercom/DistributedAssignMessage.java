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

package org.nd4j.parameterserver.distributed.messages.intercom;

import lombok.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.DistributedMessage;
import org.nd4j.parameterserver.distributed.messages.RequestMessage;

/**
 * Assign target row to specified value
 *
 * @author raver119@gmail.com
 */
@Data
@Deprecated
public class DistributedAssignMessage extends BaseVoidMessage implements DistributedMessage {
    /**
     * The only use of this message is negTable sharing.
     */
    private int index;
    private double value;
    private Integer key;
    private INDArray payload;

    protected DistributedAssignMessage() {
        super();
    }

    public DistributedAssignMessage(@NonNull Integer key, int index, double value) {
        super(6);
        this.index = index;
        this.value = value;
        this.key = key;
    }

    public DistributedAssignMessage(@NonNull Integer key, INDArray payload) {
        super(6);
        this.key = key;
        this.payload = payload;
    }

    /**
     * This method assigns specific value to either specific row, or whole array.
     * Array is identified by key
     */
    @Override
    public void processMessage() {
        if (payload != null) {
            // we're assigning array
            if (storage.arrayExists(key) && storage.getArray(key).length() == payload.length())
                storage.getArray(key).assign(payload);
            else
                storage.setArray(key, payload);
        } else {
            // we're assigning number to row
            if (index >= 0) {
                if (storage.getArray(key) == null)
                    throw new RuntimeException("Init wasn't called before for key [" + key + "]");
                storage.getArray(key).getRow(index).assign(value);
            } else
                storage.getArray(key).assign(value);
        }
    }
}
