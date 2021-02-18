/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.api.records.metadata;

import java.io.Serializable;
import java.net.URI;

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
