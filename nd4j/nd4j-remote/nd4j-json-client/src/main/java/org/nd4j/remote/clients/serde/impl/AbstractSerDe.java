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

package org.nd4j.remote.clients.serde.impl;

import lombok.NonNull;
import org.nd4j.remote.clients.serde.JsonDeserializer;
import org.nd4j.remote.clients.serde.JsonSerializer;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;

public abstract class AbstractSerDe<T> implements JsonDeserializer<T>, JsonSerializer<T> {
    protected ObjectMapper objectMapper = new ObjectMapper();


    protected String serializeClass(@NonNull T obj) {
        try {
            return objectMapper.writeValueAsString(obj);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    protected T deserializeClass(@NonNull String json, @NonNull Class<T> cls) {
        try {
            return objectMapper.readValue(json, cls);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
