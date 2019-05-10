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

package org.nd4j.parameterserver.distributed.v2.messages;

/**
 * This interface describes class responsible for keeping track of VoidMessages passed through any given node via Broadcast mechanics, to avoid duplication of messages
 * @author raver119@gmail.com
 */
public interface MessagesHistoryHolder<T> {
    /**
     * This method adds id of the message to the storage, if message is unknown
     *
     * @param id
     * @return true if it's known message, false otherwise
     */
    boolean storeIfUnknownMessageId(T id);

    /**
     * This method checks if given id was already seen before
     *
     * @param id
     * @return true if it's known message, false otherwise
     */
    boolean isKnownMessageId(T id);
}
