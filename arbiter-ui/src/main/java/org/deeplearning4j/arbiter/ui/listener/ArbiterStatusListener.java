package org.deeplearning4j.arbiter.ui.listener;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.runner.CandidateInfo;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.listener.StatusChangeType;
import org.deeplearning4j.arbiter.optimize.runner.listener.StatusListener;
import org.deeplearning4j.arbiter.ui.data.GlobalConfigPersistable;
import org.deeplearning4j.arbiter.ui.data.ModelInfoPersistable;
import org.deeplearning4j.arbiter.ui.misc.JsonMapper;

import java.util.UUID;

/**
 * Created by Alex on 20/07/2017.
 */
@Slf4j
public class ArbiterStatusListener implements StatusListener {

    private final String sessionId;
    private final StatsStorageRouter statsStorage;

    private String ocJson;

    public ArbiterStatusListener(@NonNull StatsStorageRouter statsStorage) {
        this(UUID.randomUUID().toString(), statsStorage);
    }

    public ArbiterStatusListener(@NonNull String sessionId, @NonNull StatsStorageRouter statsStorage){
        this.sessionId = sessionId;
        this.statsStorage = statsStorage;
    }

    @Override
    public void onInitialization(IOptimizationRunner r) {
        log.info("onInitialization() called");
        Persistable p = getNewStatusPersistable(r);
        statsStorage.putStaticInfo(p);
    }

    @Override
    public void onShutdown(IOptimizationRunner runner) {
        //No op?

    }

    @Override
    public void onRunnerStatusChange(StatusChangeType statusChangeType, IOptimizationRunner r) {
        log.info("onRunnerStatusChange() called");
        Persistable p = getNewStatusPersistable(r);
        statsStorage.putStaticInfo(p);
    }

    @Override
    public void onCandidateStatusChange(CandidateInfo candidateInfo, IOptimizationRunner runner, OptimizationResult<?, ?, ?> result) {

        ModelInfoPersistable p = new ModelInfoPersistable.Builder()
                .timestamp(candidateInfo.getCreatedTime())
                .sessionId(sessionId)
                .workerId(String.valueOf(candidateInfo.getIndex()))
                .modelIdx(candidateInfo.getIndex())
                .score(candidateInfo.getScore())
                .status(candidateInfo.getCandidateStatus())
                .build();

        statsStorage.putUpdate(p);
    }

    @Override
    public void onCandidateIteration(Object candidate, int iteration) {

    }


    private GlobalConfigPersistable getNewStatusPersistable(IOptimizationRunner r){
//        if(ocJson == null){
            ocJson = JsonMapper.asJson(r.getConfiguration());
//        }

        GlobalConfigPersistable p = new GlobalConfigPersistable.Builder()
                .sessionId(sessionId)
                .timestamp(System.currentTimeMillis())
                .optimizationConfigJson(ocJson)
                .candidateCounts(r.numCandidatesQueued(), r.numCandidatesCompleted(),
                        r.numCandidatesFailed(), r.numCandidatesTotal())
                .build();

        return p;
    }
}
