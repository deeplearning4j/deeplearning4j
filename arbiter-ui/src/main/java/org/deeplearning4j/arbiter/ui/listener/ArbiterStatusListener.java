package org.deeplearning4j.arbiter.ui.listener;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.runner.CandidateInfo;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.listener.StatusListener;
import org.deeplearning4j.arbiter.ui.data.GlobalConfigPersistable;
import org.deeplearning4j.arbiter.ui.data.ModelInfoPersistable;
import org.deeplearning4j.arbiter.ui.misc.JsonMapper;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.eclipse.collections.impl.list.mutable.primitive.FloatArrayList;
import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Created by Alex on 20/07/2017.
 */
@Slf4j
public class ArbiterStatusListener implements StatusListener {

    private final String sessionId;
    private final StatsStorageRouter statsStorage;

    private String ocJson;
    private long startTime = 0;

    private Map<Integer,Object> candidateStaticInfo = new ConcurrentHashMap<>();
    private Map<Integer,Pair<IntArrayList,FloatArrayList>> candidateScoreVsIter = new ConcurrentHashMap<>();

    private Map<Integer,ModelInfoPersistable> lastModelInfoPersistable = new ConcurrentHashMap<>();

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
    public void onRunnerStatusChange(IOptimizationRunner r) {
        log.info("onRunnerStatusChange() called");
        Persistable p = getNewStatusPersistable(r);
        statsStorage.putStaticInfo(p);
    }

    @Override
    public void onCandidateStatusChange(CandidateInfo candidateInfo, IOptimizationRunner runner, OptimizationResult<?, ?, ?> result) {
        ModelInfoPersistable p = lastModelInfoPersistable.get(candidateInfo.getIndex());
        if(p == null){
            p = new ModelInfoPersistable.Builder()
                    .timestamp(candidateInfo.getCreatedTime())
                    .sessionId(sessionId)
                    .workerId(String.valueOf(candidateInfo.getIndex()))
                    .modelIdx(candidateInfo.getIndex())
                    .score(candidateInfo.getScore())
                    .status(candidateInfo.getCandidateStatus())
                    .build();

            lastModelInfoPersistable.put(candidateInfo.getIndex(), p);
        }

        if(p.getScore() == null){
            p.setScore(candidateInfo.getScore());
        }

        p.setStatus(candidateInfo.getCandidateStatus());


        statsStorage.putUpdate(p);
    }

    @Override
    public void onCandidateIteration(CandidateInfo candidateInfo, Object candidate, int iteration) {

        double score;
        long numParams;
        int numLayers;
        String modelConfigJson;
        int totalNumUpdates;
        if(candidate instanceof MultiLayerNetwork){
            MultiLayerNetwork m = (MultiLayerNetwork)candidate;
            score = m.score();
            numParams = m.numParams();
            numLayers = m.getnLayers();
            modelConfigJson = m.getLayerWiseConfigurations().toJson();
            totalNumUpdates = m.getLayerWiseConfigurations().getIterationCount();
        } else if(candidate instanceof ComputationGraph) {
            ComputationGraph cg = (ComputationGraph)candidate;
            score = cg.score();
            numParams = cg.numParams();
            numLayers = cg.getNumLayers();
            modelConfigJson = cg.getConfiguration().toJson();
            totalNumUpdates = cg.getConfiguration().getIterationCount();
        } else {
            score = 0;
            numParams = 0;
            numLayers = 0;
            totalNumUpdates = 0;
            modelConfigJson = "";
        }

        int idx = candidateInfo.getIndex();

        Pair<IntArrayList, FloatArrayList> pair = candidateScoreVsIter.computeIfAbsent(idx, k -> new Pair<>(new IntArrayList(), new FloatArrayList()));

        IntArrayList iter = pair.getFirst();
        FloatArrayList scores = pair.getSecond();

        iter.add(iteration);
        scores.add((float)score);


        //TODO subsample if necessary, to max N points

        int[] iters = iter.toArray();
        float[] fScores = new float[iters.length];
        for( int i=0; i<iters.length; i++ ){
            fScores[i] = scores.get(i);
        }


        ModelInfoPersistable p = new ModelInfoPersistable.Builder()
                .timestamp(candidateInfo.getCreatedTime())
                .sessionId(sessionId)
                .workerId(String.valueOf(candidateInfo.getIndex()))
                .modelIdx(candidateInfo.getIndex())
                .score(candidateInfo.getScore())
                .status(candidateInfo.getCandidateStatus())
                .scoreVsIter(iters, fScores)
                .lastUpdateTime(System.currentTimeMillis())
                .numParameters(numParams)
                .numLayers(numLayers)
                .totalNumUpdates(totalNumUpdates)
                .paramSpaceValues(candidateInfo.getFlatParams())
                .modelConfigJson(modelConfigJson)
                .build();


        lastModelInfoPersistable.put(candidateInfo.getIndex(), p);
        statsStorage.putUpdate(p);
    }


    private GlobalConfigPersistable getNewStatusPersistable(IOptimizationRunner r){
        if(ocJson == null || this.startTime == 0L){
            //Want to update config once start time has been set
            ocJson = JsonMapper.asJson(r.getConfiguration());
            this.startTime = r.getConfiguration().getExecutionStartTime();
        }

        GlobalConfigPersistable p = new GlobalConfigPersistable.Builder()
                .sessionId(sessionId)
                .timestamp(System.currentTimeMillis())
                .optimizationConfigJson(ocJson)
                .candidateCounts(r.numCandidatesQueued(), r.numCandidatesCompleted(),
                        r.numCandidatesFailed(), r.numCandidatesTotal())
                .optimizationRunner(r.getClass().getSimpleName())
                .build();

        return p;
    }
}
