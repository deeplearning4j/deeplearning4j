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

import java.net.URI;
import java.util.Collections;
import java.util.List;
import lombok.Getter;

/**
 * Record metadata to use with JDBCRecordReader. To uniquely identify and recover a record, we use a parameterized
 * request which will be prepared with the values stored in the params attribute.
 *
 * @author Adrien Plagnol
 */
public class RecordMetaDataJdbc implements RecordMetaData {

    private final URI uri;
    @Getter
    private final String request;
    @Getter
    private final List<Object> params;
    private final Class<?> readerClass;

    public RecordMetaDataJdbc(URI uri, String request, List<? extends Object> params, Class<?> readerClass) {
        this.uri = uri;
        this.request = request;
        this.params = Collections.unmodifiableList(params);
        this.readerClass = readerClass;
    }

    @Override
    public String getLocation() {
        return this.toString();
    }

    @Override
    public URI getURI() {
        return uri;
    }

    @Override
    public Class<?> getReaderClass() {
        return readerClass;
    }

    @Override
    public String toString() {
        return "jdbcRecord(uri=" + uri +
            ", request='" + request + '\'' +
            ", parameters='" + params.toString() + '\'' +
            ", readerClass=" + readerClass +
            ')';
    }
}
