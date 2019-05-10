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

package org.deeplearning4j.spark.util.serde;

import org.apache.spark.storage.StorageLevel;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;

import java.io.IOException;

/**
 * By default: Spark storage levels don't serialize/deserialize cleanly with Jackson (i.e., we can get different results out).
 * So we'll manually control the serialization/deserialization for StorageLevel objects
 *
 * @author Alex Black
 */
public class StorageLevelDeserializer extends JsonDeserializer<StorageLevel> {
    @Override
    public StorageLevel deserialize(JsonParser jsonParser, DeserializationContext deserializationContext)
                    throws IOException, JsonProcessingException {
        JsonNode node = jsonParser.getCodec().readTree(jsonParser);
        String value = node.textValue();
        if (value == null || "null".equals(value)) {
            return null;
        }
        return StorageLevel.fromString(value);
    }
}
