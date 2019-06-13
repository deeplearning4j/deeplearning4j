package java.org.deeplearning4j.rl4j.learning.sync.qlearning.discrete;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.sync.DataManagerSyncLearningEpochListener;
import org.deeplearning4j.rl4j.util.Constants;

import org.deeplearning4j.rl4j.util.IDataManager;

@Slf4j
public class DataManagerQLearningDiscreteEpochListener extends DataManagerSyncLearningEpochListener {

    private final IHistoryProcessor historyProcessor;
    private int lastMonitor = -Constants.MONITOR_FREQ;

    public DataManagerQLearningDiscreteEpochListener(IDataManager dataManager, IHistoryProcessor historyProcessor) {
        super(dataManager);
        this.historyProcessor = historyProcessor;
    }

    @Override
    public void onBeforeEpoch(ILearning learning, int currentEpoch, int currentStep) {
        if (currentStep - lastMonitor >= Constants.MONITOR_FREQ && dataManager.isSaveData()) {
            lastMonitor = currentStep;
            int[] shape = learning.getMdp().getObservationSpace().getShape();
            historyProcessor.startMonitor(dataManager.getVideoDir() + "/video-" + currentEpoch + "-"
                    + currentStep + ".mp4", shape);
        }
    }
}
