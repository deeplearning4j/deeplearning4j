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

package org.deeplearning4j.api.storage.impl;

import lombok.AllArgsConstructor;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.StorageMetaData;

import java.util.Collection;

/**
 * A simple StatsStorageRouter that simply stores the metadata, static info and updates in the specified
 * collections. Typically used for testing.
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class CollectionStatsStorageRouter implements StatsStorageRouter {

    private Collection<StorageMetaData> metaDatas;
    private Collection<Persistable> staticInfos;
    private Collection<Persistable> updates;


    @Override
    public void putStorageMetaData(StorageMetaData storageMetaData) {
        this.metaDatas.add(storageMetaData);
    }

    @Override
    public void putStorageMetaData(Collection<? extends StorageMetaData> storageMetaData) {
        this.metaDatas.addAll(storageMetaData);
    }

    @Override
    public void putStaticInfo(Persistable staticInfo) {
        this.staticInfos.add(staticInfo);
    }

    @Override
    public void putStaticInfo(Collection<? extends Persistable> staticInfo) {
        this.staticInfos.addAll(staticInfo);
    }

    @Override
    public void putUpdate(Persistable update) {
        this.updates.add(update);
    }

    @Override
    public void putUpdate(Collection<? extends Persistable> updates) {
        this.updates.addAll(updates);
    }
}
