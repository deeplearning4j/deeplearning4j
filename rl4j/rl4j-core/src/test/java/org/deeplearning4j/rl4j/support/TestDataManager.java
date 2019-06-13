package org.deeplearning4j.rl4j.support;

import lombok.Getter;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.util.DataManager;
import org.deeplearning4j.rl4j.util.IDataManager;

import java.io.IOException;

public class TestDataManager implements IDataManager {

    private final boolean isSaveData;

    @Getter
    private int appendStatCount = 0;

    @Getter
    private int writeInfoCount = 0;

    @Getter
    private int saveCount = 0;

    public TestDataManager(boolean isSaveData) {

        this.isSaveData = isSaveData;
    }

    @Override
    public String getVideoDir() {
        return "";
    }

    @Override
    public void appendStat(DataManager.StatEntry statEntry) throws IOException {
        ++appendStatCount;
    }

    @Override
    public void writeInfo(ILearning iLearning) throws IOException {
        ++writeInfoCount;
    }

    @Override
    public void save(ILearning learning) throws IOException {
        ++saveCount;
    }

    @Override
    public boolean isSaveData() {
        return isSaveData;
    }
}
