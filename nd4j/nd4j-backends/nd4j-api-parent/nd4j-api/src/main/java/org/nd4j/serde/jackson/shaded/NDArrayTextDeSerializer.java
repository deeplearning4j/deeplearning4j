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

package org.nd4j.serde.jackson.shaded;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.node.ArrayNode;

import java.io.IOException;
import java.util.Iterator;

/**
 * @author Adam Gibson
 */

public class NDArrayTextDeSerializer extends JsonDeserializer<INDArray> {
    @Override
    public INDArray deserialize(JsonParser jp, DeserializationContext deserializationContext) throws IOException {
        JsonNode n = jp.getCodec().readTree(jp);
        return deserialize(n);
    }

    public INDArray deserialize(JsonNode n){

        //First: check for backward compatilibity (RowVectorSerializer/Deserializer)
        if(!n.has("dataType")){
            int size = n.size();
            double[] d = new double[size];
            for (int i = 0; i < size; i++) {
                d[i] = n.get(i).asDouble();
            }

            return Nd4j.create(d);
        }

        //Normal deserialize
        String dtype = n.get("dataType").asText();
        DataType dt = DataType.valueOf(dtype);
        ArrayNode shapeNode = (ArrayNode)n.get("shape");
        long[] shape = new long[shapeNode.size()];
        for( int i=0; i<shape.length; i++ ){
            shape[i] = shapeNode.get(i).asLong();
        }
        ArrayNode dataNode = (ArrayNode)n.get("data");
        Iterator<JsonNode> iter = dataNode.elements();
        int i=0;
        INDArray arr;
        switch (dt){
            case DOUBLE:
                double[] d = new double[dataNode.size()];
                while(iter.hasNext())
                    d[i++] = iter.next().asDouble();
                arr = Nd4j.create(d, shape, 'c');
                break;
            case FLOAT:
            case HALF:
                float[] f = new float[dataNode.size()];
                while(iter.hasNext())
                    f[i++] = iter.next().floatValue();
                arr = Nd4j.create(f, shape, 'c').castTo(dt);
                break;
            case LONG:
                long[] l = new long[dataNode.size()];
                while(iter.hasNext())
                    l[i++] = iter.next().longValue();
                arr = Nd4j.createFromArray(l).reshape('c', shape);
                break;
            case INT:
            case SHORT:
            case UBYTE:
                int[] a = new int[dataNode.size()];
                while(iter.hasNext())
                    a[i++] = iter.next().intValue();
                arr = Nd4j.createFromArray(a).reshape('c', shape).castTo(dt);
                break;
            case BYTE:
            case BOOL:
                byte[] b = new byte[dataNode.size()];
                while(iter.hasNext())
                    b[i++] = (byte)iter.next().intValue();
                arr = Nd4j.createFromArray(b).reshape('c', shape).castTo(dt);
                break;
            case UTF8:
                String[] s = new String[dataNode.size()];
                while(iter.hasNext())
                    s[i++] = iter.next().asText();
                arr = Nd4j.create(s).reshape('c', shape);
                break;
            case COMPRESSED:
            case UNKNOWN:
            default:
                throw new RuntimeException("Unknown datatype: " + dt);
        }
        return arr;
    }
}
