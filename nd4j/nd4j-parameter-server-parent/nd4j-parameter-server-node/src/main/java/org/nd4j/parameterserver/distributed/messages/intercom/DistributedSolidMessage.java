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

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.DistributedMessage;

/**
 * Array passed here will be shared & available on all shards.
 *
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
@Deprecated
public class DistributedSolidMessage extends BaseVoidMessage implements DistributedMessage {
    /**
     * The only use of this message is negTable sharing.
     */

    private Integer key;
    private INDArray payload;
    private boolean overwrite;

    public DistributedSolidMessage(@NonNull Integer key, @NonNull INDArray array, boolean overwrite) {
        super(5);
        this.payload = array;
        this.key = key;
        this.overwrite = overwrite;
    }

    /**
     * This method will be started in context of executor, either Shard, Client or Backup node
     */
    @Override
    public void processMessage() {
        if (overwrite)
            storage.setArray(key, payload);
        else if (!storage.arrayExists(key))
            storage.setArray(key, payload);
    }
}
