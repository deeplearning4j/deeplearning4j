package org.deeplearning4j.ui.storage;

import org.deeplearning4j.ui.storage.mapdb.MapDBStatsStorage;

import java.io.File;

/**
 * A StatsStorage implementation that stores UI data in a file for persistence.<br>
 * Can be used for multiple instances, and across multiple independent runs. Data can be loaded later in a separate
 * JVM instance by passing the same file location to both.<br>
 * Internally, uses {@link MapDBStatsStorage}
 *
 * @author Alex Black
 */
public class FileStatsStorage extends MapDBStatsStorage {

    private final File file;

    public FileStatsStorage(File f) {
        super(f);
        this.file = f;
    }

    @Override
    public String toString() {
        return "FileStatsStorage(" + file.getPath() + ")";
    }
}
