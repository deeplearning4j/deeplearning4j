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

package org.nd4j.serde.jackson.shaded;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.serde.base64.Nd4jBase64;
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;

import java.io.IOException;

/**
 * @author Alex Black
 */
public class NDArrayTextSerializer extends JsonSerializer<INDArray> {
    @Override
    public void serialize(INDArray arr, JsonGenerator jg, SerializerProvider serializerProvider)
                    throws IOException {
        jg.writeStartObject();
        jg.writeStringField("dataType", arr.dataType().toString());
        jg.writeArrayFieldStart("shape");
        for( int i=0; i<arr.rank(); i++ ){
            jg.writeNumber(arr.size(i));
        }
        jg.writeEndArray();
        jg.writeArrayFieldStart("data");

        if(arr.isView() || arr.ordering() != 'c' || !Shape.hasDefaultStridesForShape(arr) || arr.isCompressed())
            arr = arr.dup('c');

        switch (arr.dataType()){
            case DOUBLE:
                double[] d = arr.data().asDouble();
                for( double v : d )
                    jg.writeNumber(v);
                break;
            case FLOAT:
            case HALF:
                float[] f = arr.data().asFloat();
                for( float v : f )
                    jg.writeNumber(v);
                break;
            case LONG:
                long[] l = arr.data().asLong();
                for( long v : l )
                    jg.writeNumber(v);
                break;
            case INT:
            case SHORT:
            case UBYTE:
                int[] i = arr.data().asInt();
                for( int v : i )
                    jg.writeNumber(v);
                break;
            case BYTE:
            case BOOL:
                byte[] b = arr.data().asBytes();
                for( byte v : b )
                    jg.writeNumber(v);
                break;
            case UTF8:
                String[] str = new String[(int)arr.length()];
                for( int j=0; j<str.length; j++ )
                    jg.writeString(arr.getStringUnsafe(j));
                break;
            case COMPRESSED:
            case UNKNOWN:
                throw new UnsupportedOperationException("Cannot JSON serialize array with datatype: " + arr.dataType());
        }
        jg.writeEndArray();
        jg.writeEndObject();
    }
}
