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
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * By default: Spark storage levels don't serialize/deserialize cleanly with Jackson (i.e., we can get different results out).
 * So we'll manually control the serialization/deserialization for StorageLevel objects
 *
 * @author Alex Black
 */
public class StorageLevelSerializer extends JsonSerializer<StorageLevel> {

    private static final Map<StorageLevel, String> map = initMap();

    private static Map<StorageLevel, String> initMap() {
        Map<StorageLevel, String> map = new HashMap<>();
        map.put(StorageLevel.NONE(), "NONE");
        map.put(StorageLevel.DISK_ONLY(), "DISK_ONLY");
        map.put(StorageLevel.DISK_ONLY_2(), "DISK_ONLY_2");
        map.put(StorageLevel.MEMORY_ONLY(), "MEMORY_ONLY");
        map.put(StorageLevel.MEMORY_ONLY_2(), "MEMORY_ONLY_2");
        map.put(StorageLevel.MEMORY_ONLY_SER(), "MEMORY_ONLY_SER");
        map.put(StorageLevel.MEMORY_ONLY_SER_2(), "MEMORY_ONLY_SER_2");
        map.put(StorageLevel.MEMORY_AND_DISK(), "MEMORY_AND_DISK");
        map.put(StorageLevel.MEMORY_AND_DISK_2(), "MEMORY_AND_DISK_2");
        map.put(StorageLevel.MEMORY_AND_DISK_SER(), "MEMORY_AND_DISK_SER");
        map.put(StorageLevel.MEMORY_AND_DISK_SER_2(), "MEMORY_AND_DISK_SER_2");
        map.put(StorageLevel.OFF_HEAP(), "OFF_HEAP");
        return map;
    }

    @Override
    public void serialize(StorageLevel storageLevel, JsonGenerator jsonGenerator, SerializerProvider serializerProvider)
                    throws IOException, JsonProcessingException {
        //This is a little ugly, but Spark doesn't provide many options here...
        String s = null;
        if (storageLevel != null) {
            s = map.get(storageLevel);
        }
        jsonGenerator.writeString(s);
    }
}
