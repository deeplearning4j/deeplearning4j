package org.deeplearning4j.ui.storage;

import org.deeplearning4j.ui.storage.mapdb.MapDBStatsStorage;
import org.deeplearning4j.util.UIDProvider;

import java.util.UUID;

/**
 * A StatsStorage implementation that stores all data in memory. If persistence is required for the UI information,
 * use {@link FileStatsStorage} or {@link MapDBStatsStorage}.<br>
 * Internally, this implementation uses {@link MapDBStatsStorage}
 *
 * @author Alex Black
 */
public class InMemoryStatsStorage extends MapDBStatsStorage {
    private final String uid;

    public InMemoryStatsStorage(){
        super();
        String str = UUID.randomUUID().toString();
        uid = str.substring(0,Math.min(str.length(),8));
    }

    @Override
    public String toString(){
        return "InMemoryStatsStorage(uid=" + uid + ")";
    }
}
