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

import lombok.Data;

import java.net.URI;

/**
 * A RecordMetaData instance that combines multiple individual RecordMetaData instances
 *
 * @author Alex Black
 */
@Data
public class RecordMetaDataComposable implements RecordMetaData {

    private Class<?> readerClass;
    private RecordMetaData[] meta;

    public RecordMetaDataComposable(RecordMetaData... recordMetaDatas) {
        this(null, recordMetaDatas);
    }

    public RecordMetaDataComposable(Class<?> readerClass, RecordMetaData... recordMetaDatas) {
        this.readerClass = readerClass;
        this.meta = recordMetaDatas;
    }

    @Override
    public String getLocation() {
        StringBuilder sb = new StringBuilder();
        sb.append("locations(");
        boolean first = true;
        for (RecordMetaData rmd : meta) {
            if (!first)
                sb.append(",");
            sb.append(rmd.getLocation());
            first = false;
        }
        sb.append(")");
        return sb.toString();
    }

    @Override
    public URI getURI() {
        return meta[0].getURI();
    }

    @Override
    public Class<?> getReaderClass() {
        return readerClass;
    }
}
