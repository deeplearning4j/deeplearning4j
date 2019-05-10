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
