package org.deeplearning4j.ui.storage;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

import java.util.Map;

/**
 * Simple abstract in-memory storage with history option to be used across threads using UiServer
 *
 * @author raver119@gmail.com
 */
public class HistoryStorage {
    public enum TargetVersion {
        LATEST,
        ANY,
        OLDEST
    }

    public enum SortOutput {
        DESCENDING,
        ASCENDING,
        NONE
    }

    // simple storage here: Key, Version, Object
    private Table<Object, Long, Object> historyTable = HashBasedTable.create();

    private static HistoryStorage ourInstance = new HistoryStorage();

    public static HistoryStorage getInstance() {
        return ourInstance;
    }

    private HistoryStorage() {
    }

    public Object get(Object key, TargetVersion version, SortOutput sort) {
        if (version.equals(TargetVersion.LATEST)) {
            Map<Long, Object>  map = historyTable.row(key);
            if (map.size() == 1) {
                return map.values().iterator().next();
            }
        }
        return null;
    }

    public Object getLatest(Object key) {
        return get(key, TargetVersion.LATEST, SortOutput.NONE);
    }

    public void put(Object key, Long version, Object object) {
        historyTable.put(key, version, object);
    }
}
