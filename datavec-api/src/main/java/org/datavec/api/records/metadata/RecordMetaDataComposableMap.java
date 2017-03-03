/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.records.metadata;

import lombok.Data;

import java.net.URI;
import java.util.Map;

/**
 * A RecordMetaData instance that combines multiple individual RecordMetaData instances, via a {@code Map<String,RecordMetaData>}
 *
 * @author Alex Black
 */
@Data
public class RecordMetaDataComposableMap implements RecordMetaData {

    private Class<?> readerClass;
    private Map<String, RecordMetaData> meta;

    public RecordMetaDataComposableMap(Map<String, RecordMetaData> recordMetaDatas) {
        this(null, recordMetaDatas);
    }

    public RecordMetaDataComposableMap(Class<?> readerClass, Map<String, RecordMetaData> recordMetaDatas) {
        this.readerClass = readerClass;
        this.meta = recordMetaDatas;
    }

    @Override
    public String getLocation() {
        StringBuilder sb = new StringBuilder();
        sb.append("locations(");
        boolean first = true;
        for (Map.Entry<String, RecordMetaData> rmd : meta.entrySet()) {
            if (!first)
                sb.append(",");
            sb.append(rmd.getKey()).append("=");
            sb.append(rmd.getValue().getLocation());
            first = false;
        }
        sb.append(")");
        return sb.toString();
    }

    @Override
    public URI getURI() {
        String first = meta.keySet().iterator().next();
        return meta.get(first).getURI();
    }

    @Override
    public Class<?> getReaderClass() {
        return readerClass;
    }
}
