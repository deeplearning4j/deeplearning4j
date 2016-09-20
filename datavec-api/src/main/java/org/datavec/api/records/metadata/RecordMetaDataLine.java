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

import lombok.AllArgsConstructor;
import lombok.Data;
import org.apache.commons.io.FilenameUtils;

import java.net.URI;

/**
 * Created by Alex on 20/09/2016.
 */
@AllArgsConstructor @Data
public class RecordMetaDataLine implements RecordMetaData {

    private int lineNumber;
    private URI uriLocation;
    private Class<?> readerClass;


    @Override
    public String getLocation() {
        String filename;
        if(uriLocation != null){
            String str = uriLocation.toString();
            filename = FilenameUtils.getBaseName(str) + "." + FilenameUtils.getExtension(str) + " ";
        } else {
            filename = "";
        }
        return filename + "line " + lineNumber;
    }

    @Override
    public Class<?> getReaderClass() {
        return readerClass;
    }
}
