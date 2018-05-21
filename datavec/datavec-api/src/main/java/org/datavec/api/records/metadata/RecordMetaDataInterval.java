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

import lombok.AllArgsConstructor;
import lombok.Data;

import java.net.URI;

/**
 * A general-purpose RecordMetaData implementation, with two indices (long values), generally forming an interval
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Data
public class RecordMetaDataInterval implements RecordMetaData {

    private final long from;
    private final long to;
    private final URI uri;
    private Class<?> readerClass;

    public RecordMetaDataInterval(long from, long to, URI uri) {
        this(from, to, uri, null);
    }

    @Override
    public String getLocation() {
        return "interval(" + from + "," + to + ")";
    }

    @Override
    public URI getURI() {
        return uri;
    }

    @Override
    public Class<?> getReaderClass() {
        return readerClass;
    }
}
