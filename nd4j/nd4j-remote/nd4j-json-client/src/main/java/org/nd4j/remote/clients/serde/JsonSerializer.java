/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

package org.nd4j.remote.clients.serde;

/**
 * This interface describes basic JSON serializer interface used for JsonRemoteInference
 * @param <T> type of the serializable class
 *
 * @author raver119@gmail.com
 */
public interface JsonSerializer<T> {

    /**
     * This method serializes given object into JSON-string
     *
     * @param o object to be serialized
     * @return
     */
    String serialize(T o);
}
