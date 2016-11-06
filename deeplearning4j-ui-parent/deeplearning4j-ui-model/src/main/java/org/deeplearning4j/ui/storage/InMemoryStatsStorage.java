package org.deeplearning4j.ui.storage;

import org.deeplearning4j.ui.storage.mapdb.MapDBStatsStorage;

/**
 * A StatsStorage implementation that stores all data in memory. If persistence is required for the UI information,
 * use {@link FileStatsStorage} or {@link MapDBStatsStorage}.<br>
 * Internally, this implementation uses {@link MapDBStatsStorage}
 *
 * @author Alex Black
 */
public class InMemoryStatsStorage extends MapDBStatsStorage {
    public InMemoryStatsStorage(){
        super();
    }
}
