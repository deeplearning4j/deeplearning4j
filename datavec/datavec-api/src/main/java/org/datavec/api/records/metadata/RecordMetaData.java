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

import java.io.Serializable;
import java.net.URI;

/**
 * RecordMetaData includes details on the record itself - for example, the source file or line number.<br>
 * It is used in conjunction with {@link org.datavec.api.records.reader.RecordReaderMeta}.<br>
 * There are two primary uses:<br>
 * (a) Tracking where a record has come from, for debugging purposes for example<br>
 * (b) Loading the raw data again later, from the record reader<br>
 *
 * @author Alex Black
 */
public interface RecordMetaData extends Serializable {

    /**
     * Get a human-readable location for the data
     */
    String getLocation();

    /**
     * Return the URI for the source of the record
     *
     * @return The URI for the record (file, etc) - or null otherwise
     */
    URI getURI();

    /**
     * Get the class that was used to generate the record
     */
    Class<?> getReaderClass();

}
