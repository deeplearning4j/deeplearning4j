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

package org.datavec.api.transform.analysis.json;

import com.tdunning.math.stats.TDigest;
import org.apache.commons.codec.binary.Base64;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

public class TDigestDeserializer extends JsonDeserializer<TDigest> {
    @Override
    public TDigest deserialize(JsonParser jp, DeserializationContext d) throws IOException, JsonProcessingException {
        JsonNode node = (JsonNode)jp.getCodec().readTree(jp);
        String field = node.get("digest").asText();
        Base64 b = new Base64();
        byte[] bytes = b.decode(field);
        try(ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(bytes))){
            return (TDigest) ois.readObject();
        } catch (Exception e){
            throw new RuntimeException("Error deserializing TDigest object from JSON", e);
        }
    }
}
