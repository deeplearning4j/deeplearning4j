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

package org.nd4j.remote.clients.serde.impl;

import lombok.NonNull;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.ObjectMapper;

/**
 * This class provides fake JSON serializer/deserializer functionality for String.
 * It doesn't put any JSON-specific bits into actual string
 */
public class StringSerde extends AbstractSerDe<String> {

    @Override
    public String serialize(@NonNull String data) {
        return serializeClass(data);
    }

    @Override
    public String deserialize(@NonNull String json) {
        return deserializeClass(json, String.class);
    }
}
