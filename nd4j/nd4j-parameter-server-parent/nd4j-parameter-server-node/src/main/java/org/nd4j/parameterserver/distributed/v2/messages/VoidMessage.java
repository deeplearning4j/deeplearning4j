/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.parameterserver.distributed.v2.messages;

import org.agrona.concurrent.UnsafeBuffer;
import org.nd4j.common.util.SerializationUtils;

import java.io.ByteArrayInputStream;
import java.io.Serializable;

public interface VoidMessage extends Serializable {
    /**
     * This method returns unique messageId
     * @return
     */
    String getMessageId();

    /**
     * This message allows to set messageId
     */
    //void setMessageId();

    /**
     * This method returns Id of originator
     * @return
     */
    String getOriginatorId();

    /**
     * This method allows to set originator id
     * PLEASE NOTE: This method must be used only from Transport context
     */
    void setOriginatorId(String id);

    /**
     * This method serializes this VoidMessage into UnsafeBuffer
     *
     * @return
     */
    default UnsafeBuffer asUnsafeBuffer() {
        return new UnsafeBuffer(SerializationUtils.toByteArray(this));
    }

    static VoidMessage fromBytes(byte[] bytes) {
        return SerializationUtils.readObject(new ByteArrayInputStream(bytes));
    }
}
