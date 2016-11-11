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
