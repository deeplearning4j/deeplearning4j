package org.deeplearning4j.spark.impl.listeners;

import lombok.Data;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.StorageMetaData;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * Standard router for use in Spark: simply collect the data for later serialization and passing back to the master.
 *
 * @author Alex Black
 */
@Data
public class VanillaStatsStorageRouter implements StatsStorageRouter {

    private final List<StorageMetaData> storageMetaData =
                    Collections.synchronizedList(new ArrayList<StorageMetaData>());
    private final List<Persistable> staticInfo = Collections.synchronizedList(new ArrayList<Persistable>());
    private final List<Persistable> updates = Collections.synchronizedList(new ArrayList<Persistable>());

    @Override
    public void putStorageMetaData(StorageMetaData storageMetaData) {
        this.storageMetaData.add(storageMetaData);
    }

    @Override
    public void putStorageMetaData(Collection<? extends StorageMetaData> storageMetaData) {
        this.storageMetaData.addAll(storageMetaData);
    }

    @Override
    public void putStaticInfo(Persistable staticInfo) {
        this.staticInfo.add(staticInfo);
    }

    @Override
    public void putStaticInfo(Collection<? extends Persistable> staticInfo) {
        this.staticInfo.addAll(staticInfo);
    }

    @Override
    public void putUpdate(Persistable update) {
        this.updates.add(update);
    }

    @Override
    public void putUpdate(Collection<? extends Persistable> updates) {
        this.updates.addAll(updates);
    }


    public List<StorageMetaData> getStorageMetaData() {
        //We can't return synchronized lists list this for Kryo: with default config, it will fail to deserialize the
        // synchronized lists, throwing an obscure null pointer exception
        return new ArrayList<>(storageMetaData);
    }

    public List<Persistable> getStaticInfo() {
        //We can't return synchronized lists list this for Kryo: with default config, it will fail to deserialize the
        // synchronized lists, throwing an obscure null pointer exception
        return new ArrayList<>(staticInfo);
    }

    public List<Persistable> getUpdates() {
        //We can't return synchronized lists list this for Kryo: with default config, it will fail to deserialize the
        // synchronized lists, throwing an obscure null pointer exception
        return new ArrayList<>(updates);
    }
}
