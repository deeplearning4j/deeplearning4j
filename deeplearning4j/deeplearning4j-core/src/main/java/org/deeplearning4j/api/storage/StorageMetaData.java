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

package org.deeplearning4j.api.storage;

import java.io.Serializable;

/**
 * StorageMetaData: contains metadata (such at types, and arbitrary custom serializable data) for storage
 *
 * @author Alex Black
 */
public interface StorageMetaData extends Persistable {

    /**
     * Timestamp for the metadata
     */
    long getTimeStamp();

    /**
     * Session ID for the metadata
     */
    String getSessionID();

    /**
     * Type ID for the metadata
     */
    String getTypeID();

    /**
     * Worker ID for the metadata
     */
    String getWorkerID();

    /**
     * Full class name for the initialization information that will be posted. Is expected to implement {@link Persistable}.
     */
    String getInitTypeClass();

    /**
     * Full class name for the update information that will be posted. Is expected to implement {@link Persistable}.
     */
    String getUpdateTypeClass();

    /**
     * Get extra metadata, if any
     */
    Serializable getExtraMetaData();

}
