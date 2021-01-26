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
package org.deeplearning4j.nn.conf.serde.format;

import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.DataFormat;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;

import java.io.IOException;

/**
 * Simple JSON deserializer for {@link DataFormat} instances - {@link CNN2DFormat} and {@link RNNFormat}
 *
 * @author Alex Black
 */
public class DataFormatDeserializer extends JsonDeserializer<DataFormat> {
    @Override
    public DataFormat deserialize(JsonParser jp, DeserializationContext deserializationContext) throws IOException, JsonProcessingException {
        JsonNode node = jp.getCodec().readTree(jp);
        String text = node.textValue();
        switch (text){
            case "NCHW":
                return CNN2DFormat.NCHW;
            case "NHWC":
                return CNN2DFormat.NHWC;
            case "NCW":
                return RNNFormat.NCW;
            case "NWC":
                return RNNFormat.NWC;
            default:
                return null;
        }
    }
}
