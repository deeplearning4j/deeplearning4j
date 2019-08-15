package org.deeplearning4j.rl4j.learning.sync.support;

import lombok.AllArgsConstructor;
import lombok.Value;
import org.deeplearning4j.rl4j.util.IDataManager;

@AllArgsConstructor
@Value
public class MockStatEntry implements IDataManager.StatEntry {
    int epochCounter;
    int stepCounter;
    double reward;
}
