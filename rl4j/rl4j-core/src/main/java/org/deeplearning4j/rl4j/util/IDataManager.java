package org.deeplearning4j.rl4j.util;

import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.util.DataManager;

import java.io.IOException;

public interface IDataManager {
    String getVideoDir();

    void appendStat(DataManager.StatEntry statEntry) throws IOException;

    void writeInfo(ILearning iLearning) throws IOException;

    void save(ILearning learning) throws IOException;

    boolean isSaveData();
}
