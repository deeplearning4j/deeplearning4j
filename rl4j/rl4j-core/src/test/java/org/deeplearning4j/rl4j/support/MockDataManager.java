package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.util.IDataManager;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MockDataManager implements IDataManager {

    private final boolean isSaveData;
    public List<StatEntry> statEntries = new ArrayList<>();
    public int isSaveDataCallCount = 0;
    public int getVideoDirCallCount = 0;
    public int writeInfoCallCount = 0;
    public int saveCallCount = 0;

    public MockDataManager(boolean isSaveData) {
        this.isSaveData = isSaveData;
    }

    @Override
    public boolean isSaveData() {
        ++isSaveDataCallCount;
        return isSaveData;
    }

    @Override
    public String getVideoDir() {
        ++getVideoDirCallCount;
        return null;
    }

    @Override
    public void appendStat(StatEntry statEntry) throws IOException {
        statEntries.add(statEntry);
    }

    @Override
    public void writeInfo(ILearning iLearning) throws IOException {
        ++writeInfoCallCount;
    }

    @Override
    public void save(ILearning learning) throws IOException {
        ++saveCallCount;
    }
}
