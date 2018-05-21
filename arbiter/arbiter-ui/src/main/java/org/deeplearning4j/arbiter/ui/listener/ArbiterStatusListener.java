package org.deeplearning4j.arbiter.ui.listener;

import it.unimi.dsi.fastutil.floats.FloatArrayList;
import it.unimi.dsi.fastutil.ints.IntArrayList;
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
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.primitives.Pair;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A {@link StatusListener} for reporting Arbiter/DL4J optimization results to a {@link StatsStorageRouter}
 *
 * @author Alex Black
 */
@Slf4j
public class ArbiterStatusListener implements StatusListener {

    public static final int MAX_SCORE_VS_ITER_PTS = 1024;   //Above this: subsample... every 2nd, 4th, 8th etc

    private final String sessionId;
    private final StatsStorageRouter statsStorage;

    private String ocJson;
    private long startTime = 0;

    private Map<Integer,Integer> candidateScoreVsIterSubsampleFreq = new ConcurrentHashMap<>();
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
        Persistable p = getNewStatusPersistable(r);
        statsStorage.putStaticInfo(p);
    }

    @Override
    public void onShutdown(IOptimizationRunner runner) {
        //No op?

    }

    @Override
    public void onRunnerStatusChange(IOptimizationRunner r) {
        Persistable p = getNewStatusPersistable(r);
        statsStorage.putStaticInfo(p);
    }

    @Override
    public void onCandidateStatusChange(CandidateInfo candidateInfo, IOptimizationRunner runner, OptimizationResult result) {
        ModelInfoPersistable p = lastModelInfoPersistable.get(candidateInfo.getIndex());
        if(p == null){
            p = new ModelInfoPersistable.Builder()
                    .timestamp(candidateInfo.getCreatedTime())
                    .sessionId(sessionId)
                    .workerId(String.valueOf(candidateInfo.getIndex()))
                    .modelIdx(candidateInfo.getIndex())
                    .score(candidateInfo.getScore())
                    .status(candidateInfo.getCandidateStatus())
                    .exceptionStackTrace(candidateInfo.getExceptionStackTrace())
                    .build();

            lastModelInfoPersistable.put(candidateInfo.getIndex(), p);
        }

        if(p.getScore() == null){
            p.setScore(candidateInfo.getScore());
        }

        if(result != null && p.getExceptionStackTrace() == null && result.getCandidateInfo().getExceptionStackTrace() != null){
            //Update exceptions that may have occurred since earlier model info instance
            p.setExceptionStackTrace(result.getCandidateInfo().getExceptionStackTrace());
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

        //Do we need subsampling to avoid having too many data points?
        int subsamplingFreq = candidateScoreVsIterSubsampleFreq.computeIfAbsent(idx, k -> 1);
        if(iteration / subsamplingFreq > MAX_SCORE_VS_ITER_PTS){
            //Double subsampling frequency and re-parse data
            subsamplingFreq *= 2;
            candidateScoreVsIterSubsampleFreq.put(idx, subsamplingFreq);

            IntArrayList newIter = new IntArrayList();
            FloatArrayList newScores = new FloatArrayList();
            for( int i=0; i<iter.size(); i++ ){
                int it = iter.get(i);
                if(it % subsamplingFreq == 0){
                    newIter.add(it);
                    newScores.add(scores.get(i));
                }
            }

            iter = newIter;
            scores = newScores;
            candidateScoreVsIter.put(idx, new Pair<>(iter, scores));
        }

        if(iteration % subsamplingFreq == 0) {
            iter.add(iteration);
            scores.add((float) score);
        }


        int[] iters = iter.toIntArray();
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
                .exceptionStackTrace(candidateInfo.getExceptionStackTrace())
                .build();


        lastModelInfoPersistable.put(candidateInfo.getIndex(), p);
        statsStorage.putUpdate(p);
    }


    private GlobalConfigPersistable getNewStatusPersistable(IOptimizationRunner r){
//        if(ocJson == null || this.startTime == 0L){
//            //Want to update config once start time has been set
//            ocJson = JsonMapper.asJson(r.getConfiguration());
//            this.startTime = r.getConfiguration().getExecutionStartTime();
//        }
        //TODO: cache global config, but we don't want to have outdated info (like uninitialized termination conditions)

        ocJson = JsonMapper.asJson(r.getConfiguration());

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
