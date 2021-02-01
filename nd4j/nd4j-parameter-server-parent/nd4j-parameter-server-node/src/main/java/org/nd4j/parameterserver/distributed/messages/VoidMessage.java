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

package org.nd4j.parameterserver.distributed.messages;

import org.agrona.concurrent.UnsafeBuffer;
import org.apache.commons.io.input.ClassLoaderObjectInputStream;
import org.nd4j.common.config.ND4JClassLoading;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.Storage;
import org.nd4j.parameterserver.distributed.logic.completion.Clipboard;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;
import org.nd4j.parameterserver.distributed.transport.Transport;

import java.io.ByteArrayInputStream;
import java.io.ObjectInputStream;
import java.io.Serializable;

/**
 * @author raver119@gmail.com
 */
@Deprecated
public interface VoidMessage extends Serializable {

    void setTargetId(short id);

    short getTargetId();

    long getTaskId();

    int getMessageType();

    long getOriginatorId();

    void setOriginatorId(long id);

    byte[] asBytes();

    UnsafeBuffer asUnsafeBuffer();

    @SuppressWarnings("unchecked")
    static <T extends VoidMessage> T fromBytes(byte[] array) {
        ClassLoader classloader = ND4JClassLoading.getNd4jClassloader();

        try (ByteArrayInputStream bis = new ByteArrayInputStream(array);
             ObjectInputStream ois = new ClassLoaderObjectInputStream(classloader, bis)) {
            return (T) ois.readObject();
        } catch (Exception objectReadException) {
            throw new RuntimeException(objectReadException);
        }
    }

    /**
     * This method initializes message for further processing
     */
    void attachContext(VoidConfiguration voidConfiguration, TrainingDriver<? extends TrainingMessage> trainer,
                    Clipboard clipboard, Transport transport, Storage storage, NodeRole role, short shardIndex);

    void extractContext(BaseVoidMessage message);

    /**
     * This method will be started in context of executor, either Shard, Client or Backup node
     */
    void processMessage();

    boolean isJoinSupported();

    boolean isBlockingMessage();

    void joinMessage(VoidMessage message);

    int getRetransmitCount();

    void incrementRetransmitCount();
}
