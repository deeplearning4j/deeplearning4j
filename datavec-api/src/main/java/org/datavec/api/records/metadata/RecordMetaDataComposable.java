/*
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

import java.net.URI;

/**
 * Created by Alex on 27/09/2016.
 */
public class RecordMetaDataComposable implements RecordMetaData {

    private RecordMetaData[] meta;

    public RecordMetaDataComposable(RecordMetaData... recordMetaDatas){
        this.meta = recordMetaDatas;
    }

    @Override
    public String getLocation() {
        StringBuilder sb = new StringBuilder();
        sb.append("locations(");
        boolean first = true;
        for(RecordMetaData rmd : meta){
            if(!first) sb.append(",");
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
        return null;    //TODO
    }
}
