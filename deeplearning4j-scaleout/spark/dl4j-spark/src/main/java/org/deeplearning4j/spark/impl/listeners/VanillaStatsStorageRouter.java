package org.deeplearning4j.spark.impl.listeners;

import lombok.Data;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.StatsStorageRouterProvider;
import org.deeplearning4j.api.storage.StorageMetaData;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by Alex on 12/10/2016.
 */
@Data
public class VanillaStatsStorageRouter implements StatsStorageRouter {

    private final List<StorageMetaData> storageMetaData = Collections.synchronizedList(new ArrayList<StorageMetaData>());
    private final List<Persistable> staticInfo = Collections.synchronizedList(new ArrayList<Persistable>());
    private final List<Persistable> updates = Collections.synchronizedList(new ArrayList<Persistable>());

    @Override
    public void putStorageMetaData(StorageMetaData storageMetaData) {
        this.storageMetaData.add(storageMetaData);
    }

    @Override
    public void putStaticInfo(Persistable staticInfo) {
        this.staticInfo.add(staticInfo);
    }

    @Override
    public void putUpdate(Persistable update) {
        this.updates.add(update);
    }
}
