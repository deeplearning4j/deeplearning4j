package org.deeplearning4j.rl4j.learning.async;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.ILearning;

import org.deeplearning4j.rl4j.util.IDataManager;

import java.io.IOException;

@Slf4j
@AllArgsConstructor
public class DataManagerAsyncTrainingListener implements AsyncTrainingListener {

    private IDataManager dataManager;

    @Override
    public void onTrainingProgress(ILearning learning) {
        try {
            dataManager.writeInfo(learning);
        } catch (IOException e) {
            log.error("WriteInfo failed.", e);
            e.printStackTrace();
        }
    }
}
