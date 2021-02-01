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

package org.deeplearning4j.core.storage;


import java.util.Collection;

public interface StatsStorageRouter {


    /**
     * Method to store some additional metadata for each session. Idea: record the classes used to
     * serialize and deserialize the static info and updates (as a class name).
     * This is mainly used for debugging and validation.
     *
     * @param storageMetaData Storage metadata to store
     */
    void putStorageMetaData(StorageMetaData storageMetaData); //TODO error handling

    void putStorageMetaData(Collection<? extends StorageMetaData> storageMetaData);

    /**
     * Static info: reported once per session, upon initialization
     *
     * @param staticInfo    Static info to store
     */
    void putStaticInfo(Persistable staticInfo); //TODO error handling

    /**
     * Static info: reported once per session, upon initialization
     *
     * @param staticInfo    Static info to store
     */
    void putStaticInfo(Collection<? extends Persistable> staticInfo);

    /**
     * Updates: stored multiple times per session (periodically, for example)
     *
     * @param update    Update info to store
     */
    void putUpdate(Persistable update); //TODO error handling

    /**
     * Updates: stored multiple times per session (periodically, for example)
     *
     * @param updates    Update info to store
     */
    void putUpdate(Collection<? extends Persistable> updates);

}
