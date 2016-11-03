package org.deeplearning4j.ui.module.train;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.api.storage.StatsStorageListener;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.*;
import org.deeplearning4j.ui.i18n.I18NProvider;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.stats.api.Histogram;
import org.deeplearning4j.ui.stats.api.StatsInitializationReport;
import org.deeplearning4j.ui.stats.api.StatsReport;
import org.deeplearning4j.ui.stats.api.StatsType;
import org.deeplearning4j.ui.views.html.training.*;
import play.libs.Json;
import play.mvc.Result;
import play.mvc.Results;

import java.text.DateFormat;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.*;

import static play.mvc.Results.ok;
import static play.mvc.Results.redirect;

/**
 * Created by Alex on 14/10/2016.
 */
@Slf4j
public class TrainModule implements UIModule {

    private static final DecimalFormat df2 = new DecimalFormat("#.00");
    DateFormat dateFormat = new SimpleDateFormat("yyyy-mm-dd HH:mm:ss");
    private Map<String, StatsStorage> knownSessionIDs = new LinkedHashMap<>();
    private String currentSessionID;


    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.singletonList(StatsListener.TYPE_ID);
    }

    @Override
    public List<Route> getRoutes() {
        Route r = new Route("/train", HttpMethod.GET, FunctionType.Supplier, () -> redirect("/train/overview"));
        Route r2 = new Route("/train/overview", HttpMethod.GET, FunctionType.Supplier, () -> ok(TrainingOverview.apply(I18NProvider.getInstance())));
        Route r2a = new Route("/train/overview/data", HttpMethod.GET, FunctionType.Supplier, this::getOverviewData);
        Route r3 = new Route("/train/model", HttpMethod.GET, FunctionType.Supplier, () -> ok(TrainingModel.apply(I18NProvider.getInstance())));
        Route r3a = new Route("/train/model/graph", HttpMethod.GET, FunctionType.Supplier, this::getModelGraph);
        Route r3b = new Route("/train/model/data/:layerId", HttpMethod.GET, FunctionType.Function, this::getModelData);
        Route r4 = new Route("/train/system", HttpMethod.GET, FunctionType.Supplier, () -> ok(TrainingSystem.apply(I18NProvider.getInstance())));
        Route r4a = new Route("/train/system/data", HttpMethod.GET, FunctionType.Supplier, this::getSystemData);
        Route r5 = new Route("/train/help", HttpMethod.GET, FunctionType.Supplier, () -> ok(TrainingHelp.apply(I18NProvider.getInstance())));
        Route r6 = new Route("/train/sessions/current", HttpMethod.GET, FunctionType.Supplier, () -> ok(currentSessionID == null ? "" : currentSessionID));
        Route r6a = new Route("/train/sessions/all", HttpMethod.GET, FunctionType.Supplier, this::listSessions);
        Route r6b = new Route("/train/sessions/info", HttpMethod.GET, FunctionType.Supplier, this::sessionInfo);
        Route r6c = new Route("/train/sessions/set/:to", HttpMethod.GET, FunctionType.Function, this::setSession);

        return Arrays.asList(r, r2, r2a, r3, r3a, r3b, r4, r4a, r5, r6, r6a, r6b, r6c);
    }

    @Override
    public void reportStorageEvents(StatsStorage statsStorage, Collection<StatsStorageEvent> events) {
        for (StatsStorageEvent sse : events) {
            if (sse.getEventType() == StatsStorageListener.EventType.PostStaticInfo && StatsListener.TYPE_ID.equals(sse.getTypeID())) {
                knownSessionIDs.put(sse.getSessionID(), statsStorage);
            }
        }

        if (currentSessionID == null) getDefaultSession();
    }

    @Override
    public synchronized void onAttach(StatsStorage statsStorage) {
        for (String sessionID : statsStorage.listSessionIDs()) {
            for (String typeID : statsStorage.listTypeIDsForSession(sessionID)) {
                if (!StatsListener.TYPE_ID.equals(typeID)) continue;
                knownSessionIDs.put(sessionID, statsStorage);
            }
        }

        if (currentSessionID == null) getDefaultSession();
    }

    @Override
    public void onDetach(StatsStorage statsStorage) {
        //TODO
    }

    private void getDefaultSession() {
        if (currentSessionID != null) return;

        //TODO handle multiple workers, etc
        long mostRecentTime = Long.MIN_VALUE;
        String sessionID = null;
        for (Map.Entry<String, StatsStorage> entry : knownSessionIDs.entrySet()) {
            List<Persistable> staticInfos = entry.getValue().getAllStaticInfos(entry.getKey(), StatsListener.TYPE_ID);
            if (staticInfos == null || staticInfos.size() == 0) continue;
            Persistable p = staticInfos.get(0);
            long thisTime = p.getTimeStamp();
            if (thisTime > mostRecentTime) {
                mostRecentTime = thisTime;
                sessionID = entry.getKey();
            }
        }

        if (sessionID != null) {
            currentSessionID = sessionID;
        }
    }

    private Result listSessions() {
        return Results.ok(Json.toJson(knownSessionIDs.keySet()));
    }

    private Result sessionInfo() {
        //Display, for each session: session ID, start time, number of workers, last update
        Map<String, Object> dataEachSession = new HashMap<>();
        for (Map.Entry<String, StatsStorage> entry : knownSessionIDs.entrySet()) {
            Map<String, Object> dataThisSession = new HashMap<>();
            String sid = entry.getKey();
            StatsStorage ss = entry.getValue();
            List<String> workerIDs = ss.listWorkerIDsForSessionAndType(sid, StatsListener.TYPE_ID);
            int workerCount = (workerIDs == null ? 0 : workerIDs.size());
            List<Persistable> staticInfo = ss.getAllStaticInfos(sid, StatsListener.TYPE_ID);
            long initTime = Long.MAX_VALUE;
            if (staticInfo != null) {
                for (Persistable p : staticInfo) {
                    initTime = Math.min(p.getTimeStamp(), initTime);
                }
            }

            long lastUpdateTime = Long.MIN_VALUE;
            List<Persistable> lastUpdatesAllWorkers = ss.getLatestUpdateAllWorkers(sid, StatsListener.TYPE_ID);
            for (Persistable p : lastUpdatesAllWorkers) {
                lastUpdateTime = Math.max(lastUpdateTime, p.getTimeStamp());
            }

            dataThisSession.put("numWorkers", workerCount);
            dataThisSession.put("initTime", initTime == Long.MAX_VALUE ? "" : initTime);
            dataThisSession.put("lastUpdate", lastUpdateTime == Long.MIN_VALUE ? "" : lastUpdateTime);

            //Model info: type, # layers, # params...
            if (staticInfo != null && staticInfo.size() > 0) {
                StatsInitializationReport sr = (StatsInitializationReport) staticInfo.get(0);
                String modelClassName = sr.getModelClassName();
                if (modelClassName.endsWith("MultiLayerNetwork")) {
                    modelClassName = "MultiLayerNetwork";
                } else if (modelClassName.endsWith("ComputationGraph")) {
                    modelClassName = "ComputationGraph";
                }
                int numLayers = sr.getModelNumLayers();
                long numParams = sr.getModelNumParams();

                dataThisSession.put("modelType", modelClassName);
                dataThisSession.put("numLayers", numLayers);
                dataThisSession.put("numParams", numParams);
            } else {
                dataThisSession.put("modelType", "");
                dataThisSession.put("numLayers", "");
                dataThisSession.put("numParams", "");
            }

            dataEachSession.put(sid, dataThisSession);
        }

        return ok(Json.toJson(dataEachSession));
    }

    private Result setSession(String newSessionID) {
        if (knownSessionIDs.containsKey(newSessionID)) {
            currentSessionID = newSessionID;
            return ok();
        } else {
            return Results.badRequest("Unknown session ID: " + newSessionID);
        }
    }

    private Result getOverviewData() {
        I18N i18N = I18NProvider.getInstance();

        boolean noData = currentSessionID == null;
        //First pass (optimize later): query all data...

        StatsStorage ss = (noData ? null : knownSessionIDs.get(currentSessionID));

        //TODO HANDLE MULTIPLE WORKERS (SPARK)
        String wid = null;
        if (!noData) {
            List<String> workerIDs = ss.listWorkerIDsForSession(currentSessionID);
            if (workerIDs == null || workerIDs.size() == 0) noData = true;
            else {
                wid = workerIDs.get(0);
            }
        }

        List<Integer> scoresIterCount = new ArrayList<>();
        List<Double> scores = new ArrayList<>();

        Map<String, Object> result = new HashMap<>();
        result.put("scores", scores);
        result.put("scoresIter", scoresIterCount);

        //Get scores info
        List<Persistable> updates = (noData ? null : ss.getAllUpdatesAfter(currentSessionID, StatsListener.TYPE_ID, wid, 0));

        //Collect update ratios for weights
        //Collect standard deviations: activations, gradients, updates
        Map<String,List<Double>> updateRatios = new HashMap<>();    //Mean magnitude (updates) / mean magnitude (parameters)
        result.put("updateRatios", updateRatios);

        Map<String,List<Double>> stdevActivations = new HashMap<>();
        Map<String,List<Double>> stdevGradients = new HashMap<>();
        Map<String,List<Double>> stdevUpdates = new HashMap<>();
        result.put("stdevActivations",stdevActivations);
        result.put("stdevGradients", stdevGradients);
        result.put("stdevUpdates", stdevUpdates);

        if(!noData){
            Persistable u = updates.get(0);
            if (u instanceof StatsReport){
                StatsReport sp = (StatsReport)u;
                Map<String,Double> map = sp.getMeanMagnitudes(StatsType.Parameters);
                if(map != null){
                    for(String s : map.keySet()){
                        if(!s.toLowerCase().endsWith("w")) continue;   //TODO: more robust "weights only" approach...
                        updateRatios.put(s,new ArrayList<>());
                    }
                }

                Map<String,Double> stdGrad = sp.getStdev(StatsType.Gradients);
                if(stdGrad != null){
                    for(String s : stdGrad.keySet()){
                        if(!s.toLowerCase().endsWith("w")) continue; //TODO: more robust "weights only" approach...
                        stdevGradients.put(s, new ArrayList<>());
                    }
                }

                Map<String,Double> stdUpdate = sp.getStdev(StatsType.Updates);
                if(stdUpdate != null){
                    for(String s : stdUpdate.keySet()){
                        if(!s.toLowerCase().endsWith("w")) continue;    //TODO: more robust "weights only" approach...
                        stdevUpdates.put(s, new ArrayList<>());
                    }
                }


                Map<String,Double> stdAct = sp.getStdev(StatsType.Activations);
                if(stdAct != null){
                    for(String s : stdAct.keySet()){
                        stdevActivations.put(s, new ArrayList<>());
                    }
                }
            }
        }

        StatsReport last = null;
        if (!noData) {
            double lastScore;
            for (Persistable u : updates) {
                if (!(u instanceof StatsReport)) continue;
                last = (StatsReport) u;
                int iterCount = last.getIterationCount();
                scoresIterCount.add(iterCount);
                lastScore = last.getScore();
                scores.add(lastScore);

                //Update ratios: mean magnitudes(updates) / mean magnitudes (parameters)
                Map<String,Double> updateMM = last.getMeanMagnitudes(StatsType.Updates);
                Map<String,Double> paramMM = last.getMeanMagnitudes(StatsType.Parameters);
                if(updateMM != null && paramMM != null && updateMM.size() > 0 && paramMM.size() > 0){
                    for(String s : updateRatios.keySet()){
                        List<Double> ratioHistory = updateRatios.get(s);
                        double currUpdate = updateMM.get(s);
                        double currParam = paramMM.get(s);
                        double ratio = currUpdate / currParam;
                        ratioHistory.add(ratio);
                    }
                }

                //Standard deviations: gradients, updates, activations
                Map<String,Double> stdGrad = last.getStdev(StatsType.Gradients);
                Map<String,Double> stdUpd = last.getStdev(StatsType.Updates);
                Map<String,Double> stdAct = last.getStdev(StatsType.Activations);

                if(stdGrad != null){
                    for(String s : stdevGradients.keySet()){
                        double d = stdGrad.get(s);
                        stdevGradients.get(s).add(d);
                    }
                }
                if(stdUpd != null){
                    for(String s : stdevUpdates.keySet()){
                        double d = stdUpd.get(s);
                        stdevUpdates.get(s).add(d);
                    }
                }
                if(stdAct != null){
                    for(String s : stdevActivations.keySet()){
                        double d = stdAct.get(s);
                        stdevActivations.get(s).add(d);
                    }
                }
            }
        }

        //----- Performance Info -----

        //TODO reuse?
        String[][] perfInfo = new String[][]{
                {i18N.getMessage("train.overview.perftable.startTime"), ""},
                {i18N.getMessage("train.overview.perftable.totalRuntime"), ""},
                {i18N.getMessage("train.overview.perftable.lastUpdate"), ""},
                {i18N.getMessage("train.overview.perftable.totalParamUpdates"), ""},
                {i18N.getMessage("train.overview.perftable.updatesPerSec"), ""},
                {i18N.getMessage("train.overview.perftable.examplesPerSec"), ""}
        };

        if (last != null) {
            perfInfo[2][1] = String.valueOf(dateFormat.format(new Date(last.getTimeStamp())));
            perfInfo[3][1] = String.valueOf(last.getTotalMinibatches());
            perfInfo[4][1] = String.valueOf(df2.format(last.getMinibatchesPerSecond()));
            perfInfo[5][1] = String.valueOf(df2.format(last.getExamplesPerSecond()));
        }

        result.put("perf", perfInfo);


        // ----- Model Info -----
        String[][] modelInfo = new String[][]{
                {i18N.getMessage("train.overview.modeltable.modeltype"), ""},
                {i18N.getMessage("train.overview.modeltable.nLayers"), ""},
                {i18N.getMessage("train.overview.modeltable.nParams"), ""}
        };
        if (!noData) {
            Persistable p = ss.getStaticInfo(currentSessionID, StatsListener.TYPE_ID, wid);
            if (p != null) {
                StatsInitializationReport initReport = (StatsInitializationReport) p;
                int nLayers = initReport.getModelNumLayers();
                long numParams = initReport.getModelNumParams();
                String className = initReport.getModelClassName();

                String modelType;
                if (className.endsWith("MultiLayerNetwork")) {
                    modelType = "MultiLayerNetwork";
                } else if (className.endsWith("ComputationGraph")) {
                    modelType = "ComputationGraph";
                } else {
                    modelType = className;
                }

                modelInfo[0][1] = modelType;
                modelInfo[1][1] = String.valueOf(nLayers);
                modelInfo[2][1] = String.valueOf(numParams);
            }
        }

        result.put("model", modelInfo);

        return Results.ok(Json.toJson(result));
    }

    private Result getModelGraph() {


        boolean noData = currentSessionID == null;
        StatsStorage ss = (noData ? null : knownSessionIDs.get(currentSessionID));
        List<Persistable> allStatic = (noData ? Collections.EMPTY_LIST : ss.getAllStaticInfos(currentSessionID, StatsListener.TYPE_ID));

        if (allStatic.size() == 0) {
            return ok();
        }

        TrainModuleUtils.GraphInfo gi = getGraphInfo();
        if(gi == null) return ok();
        return ok(Json.toJson(gi));
    }

    private TrainModuleUtils.GraphInfo getGraphInfo(){
        Pair<MultiLayerConfiguration,ComputationGraphConfiguration> conf = getConfig();
        if(conf == null){
            return null;
        }

        if(conf.getFirst() != null){
            return TrainModuleUtils.buildGraphInfo(conf.getFirst());
        } else if(conf.getSecond() != null){
            return TrainModuleUtils.buildGraphInfo(conf.getSecond());
        } else {
            return null;
        }
    }

    private Pair<MultiLayerConfiguration,ComputationGraphConfiguration> getConfig(){
        boolean noData = currentSessionID == null;
        StatsStorage ss = (noData ? null : knownSessionIDs.get(currentSessionID));
        List<Persistable> allStatic = (noData ? Collections.EMPTY_LIST : ss.getAllStaticInfos(currentSessionID, StatsListener.TYPE_ID));
        if(allStatic.size() == 0) return null;

        StatsInitializationReport p = (StatsInitializationReport) allStatic.get(0);
        String modelClass = p.getModelClassName();
        String config = p.getModelConfigJson();

        if (modelClass.endsWith("MultiLayerNetwork")) {
            MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(config);
            return new Pair<>(conf,null);
        } else if (modelClass.endsWith("ComputationGraph")) {
            ComputationGraphConfiguration conf = ComputationGraphConfiguration.fromJson(config);
            return new Pair<>(null,conf);
        }
        return null;
    }


    private Result getModelData(String str) {
        int layerIdx = Integer.parseInt(str);   //TODO validation
        I18N i18N = I18NProvider.getInstance();

        //Model info for layer

        boolean noData = currentSessionID == null;
        //First pass (optimize later): query all data...

        StatsStorage ss = (noData ? null : knownSessionIDs.get(currentSessionID));

        //TODO HANDLE MULTIPLE WORKERS (SPARK)
        String wid = null;
        if (!noData) {
            List<String> workerIDs = ss.listWorkerIDsForSession(currentSessionID);
            if (workerIDs == null || workerIDs.size() == 0) noData = true;
            else {
                wid = workerIDs.get(0);
            }
        }


        Map<String, Object> result = new HashMap<>();

        Pair<MultiLayerConfiguration,ComputationGraphConfiguration> conf = getConfig();
        if(conf == null){
            return ok(Json.toJson(result));
        }

        TrainModuleUtils.GraphInfo gi = getGraphInfo();
        if(gi == null){
            return ok(Json.toJson(result));
        }


        // Get static layer info
        String[][] layerInfoTable = getLayerInfoTable(layerIdx, gi, i18N, noData, ss, wid);

        result.put("layerInfo", layerInfoTable);

        //Get mean magnitudes line chart
        List<Persistable> updates = (noData ? null : ss.getAllUpdatesAfter(currentSessionID, StatsListener.TYPE_ID, wid, 0));
        Pair<List<Integer>, Map<String, List<Double>>> meanMagnitudes = getLayerMeanMagnitudes(layerIdx, gi, updates, conf.getFirst() != null);
        Map<String, Object> mmRatioMap = new HashMap<>();
        mmRatioMap.put("layerParamNames", meanMagnitudes.getSecond().keySet());
        mmRatioMap.put("iterCounts", meanMagnitudes.getFirst());
        mmRatioMap.put("ratios", meanMagnitudes.getSecond());
        result.put("meanMagRatio", mmRatioMap);

        //Get activations line chart for layer

        Triple<int[], float[], float[]> activationsData = getLayerActivations(layerIdx, gi, updates, conf.getFirst(), conf.getSecond());
        Map<String, Object> activationMap = new HashMap<>();
        activationMap.put("iterCount", activationsData.getFirst());
        activationMap.put("mean", activationsData.getSecond());
        activationMap.put("stdev", activationsData.getThird());
        result.put("activations", activationMap);

        //Get learning rate vs. time chart for layer
        Map<String, Object> lrs = getLayerLearningRates(layerIdx, gi, updates);
        result.put("learningRates", lrs);

        //Parameters histogram data
        Persistable lastUpdate = (updates != null && updates.size() > 0 ? updates.get(updates.size() - 1) : null);
        Map<String, Object> paramHistograms = getHistograms(layerIdx, gi, StatsType.Parameters, lastUpdate);
        result.put("paramHist", paramHistograms);

        //Updates histogram data
        Map<String, Object> updateHistograms = getHistograms(layerIdx, gi, StatsType.Updates, lastUpdate);
        result.put("updateHist", updateHistograms);

        return ok(Json.toJson(result));
    }

    public Result getSystemData() {
        I18N i18n = I18NProvider.getInstance();

        //First: get the MOST RECENT update...
        //Then get all updates from most recent - 5 minutes -> TODO make this configurable...

        boolean noData = currentSessionID == null;
        StatsStorage ss = (noData ? null : knownSessionIDs.get(currentSessionID));

        List<Persistable> allStatic = (noData ? Collections.EMPTY_LIST : ss.getAllStaticInfos(currentSessionID, StatsListener.TYPE_ID));
        List<Persistable> latestUpdates = (noData ? Collections.EMPTY_LIST : ss.getLatestUpdateAllWorkers(currentSessionID, StatsListener.TYPE_ID));

        long lastUpdateTime = -1;
        if (latestUpdates == null || latestUpdates.size() == 0) {
            noData = true;
        } else {
            for (Persistable p : latestUpdates) {
                lastUpdateTime = Math.max(lastUpdateTime, p.getTimeStamp());
            }
        }

        long fromTime = lastUpdateTime - 5 * 60 * 1000; //TODO Make configurable
        List<Persistable> lastNMinutes = (noData ? null : ss.getAllUpdatesAfter(currentSessionID, StatsListener.TYPE_ID, fromTime));

        Map<String, Object> mem = getMemory(allStatic, lastNMinutes, i18n);
        Pair<Map<String, Object>, Map<String, Object>> hwSwInfo = getHardwareSoftwareInfo(allStatic, i18n);

        Map<String, Object> ret = new HashMap<>();
        ret.put("memory", mem);
        ret.put("hardware", hwSwInfo.getFirst());
        ret.put("software", hwSwInfo.getSecond());


        return ok(Json.toJson(ret));
    }

    private static String getLayerType(Layer layer) {
        String layerType = "n/a";
        if (layer != null) {
            try {
                layerType = layer.getClass().getSimpleName().replaceAll("Layer$", "");
            } catch (Exception e) {
            }
        }
        return layerType;
    }

    private String[][] getLayerInfoTable(int layerIdx, TrainModuleUtils.GraphInfo gi, I18N i18N, boolean noData, StatsStorage ss, String wid) {
        List<String[]> layerInfoRows = new ArrayList<>();
        layerInfoRows.add(new String[]{i18N.getMessage("train.model.layerinfotable.layerName"), gi.getVertexNames().get(layerIdx)});
        layerInfoRows.add(new String[]{i18N.getMessage("train.model.layerinfotable.layerType"), ""});

        if (!noData) {
            Persistable p = ss.getStaticInfo(currentSessionID, StatsListener.TYPE_ID, wid);
            if (p != null) {
                StatsInitializationReport initReport = (StatsInitializationReport) p;
                String configJson = initReport.getModelConfigJson();
                String modelClass = initReport.getModelClassName();

                //TODO error handling...
                String layerType = "";
                Layer layer = null;
                NeuralNetConfiguration nnc = null;
                if (modelClass.endsWith("MultiLayerNetwork")) {
                    MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(configJson);
                    int confIdx = layerIdx-1;   //-1 because of input
                    if (confIdx >= 0) {
                        nnc = conf.getConf(confIdx);
                        layer = nnc.getLayer();
                    } else {
                        //Input layer
                        layerType = "Input";
                    }
                } else if (modelClass.endsWith("ComputationGraph")) {
                    ComputationGraphConfiguration conf = ComputationGraphConfiguration.fromJson(configJson);

                    String vertexName = gi.getVertexNames().get(layerIdx);

                    Map<String, GraphVertex> vertices = conf.getVertices();
                    if (vertices.containsKey(vertexName) && vertices.get(vertexName) instanceof LayerVertex) {
                        LayerVertex lv = (LayerVertex) vertices.get(vertexName);
                        nnc = lv.getLayerConf();
                        layer = nnc.getLayer();
                    } else if(conf.getNetworkInputs().contains(vertexName)){
                        layerType = "Input";
                    } else {
                        GraphVertex gv = conf.getVertices().get(vertexName);
                        if(gv != null){
                            layerType = gv.getClass().getSimpleName();
                        }
                    }
                }

                if(layer != null) {
                    layerType = getLayerType(layer);
                }

                if (layer != null) {
                    String activationFn = null;
                    if (layer instanceof FeedForwardLayer) {
                        FeedForwardLayer ffl = (FeedForwardLayer) layer;
                        layerInfoRows.add(new String[]{i18N.getMessage("train.model.layerinfotable.layerNIn"), String.valueOf(ffl.getNIn())});
                        layerInfoRows.add(new String[]{i18N.getMessage("train.model.layerinfotable.layerSize"), String.valueOf(ffl.getNOut())});
                        activationFn = layer.getActivationFunction();
                    }
                    int nParams = layer.initializer().numParams(nnc, true);
                    layerInfoRows.add(new String[]{i18N.getMessage("train.model.layerinfotable.layerNParams"), String.valueOf(nParams)});
                    if (nParams > 0) {
                        WeightInit wi = layer.getWeightInit();
                        String str = wi.toString();
                        if (wi == WeightInit.DISTRIBUTION) {
                            str += layer.getDist();
                        }
                        layerInfoRows.add(new String[]{i18N.getMessage("train.model.layerinfotable.layerWeightInit"), str});

                        Updater u = layer.getUpdater();
                        String us = (u == null ? "" : u.toString());
                        layerInfoRows.add(new String[]{i18N.getMessage("train.model.layerinfotable.layerUpdater"), us});

                        //TODO: Maybe L1/L2, dropout, updater-specific values etc
                    }

                    if (layer instanceof ConvolutionLayer || layer instanceof SubsamplingLayer) {
                        int[] kernel;
                        int[] stride;
                        int[] padding;
                        if (layer instanceof ConvolutionLayer) {
                            ConvolutionLayer cl = (ConvolutionLayer) layer;
                            kernel = cl.getKernelSize();
                            stride = cl.getStride();
                            padding = cl.getPadding();
                        } else {
                            SubsamplingLayer ssl = (SubsamplingLayer) layer;
                            kernel = ssl.getKernelSize();
                            stride = ssl.getStride();
                            padding = ssl.getPadding();
                            activationFn = null;
                            layerInfoRows.add(new String[]{i18N.getMessage("train.model.layerinfotable.layerSubsamplingPoolingType"), ssl.getPoolingType().toString()});
                        }
                        layerInfoRows.add(new String[]{i18N.getMessage("train.model.layerinfotable.layerCnnKernel"), Arrays.toString(kernel)});
                        layerInfoRows.add(new String[]{i18N.getMessage("train.model.layerinfotable.layerCnnStride"), Arrays.toString(stride)});
                        layerInfoRows.add(new String[]{i18N.getMessage("train.model.layerinfotable.layerCnnPadding"), Arrays.toString(padding)});
                    }

                    if (activationFn != null) {
                        layerInfoRows.add(new String[]{i18N.getMessage("train.model.layerinfotable.layerActivationFn"), activationFn});
                    }
                }
                layerInfoRows.get(1)[1] = layerType;
            }
        }

        return layerInfoRows.toArray(new String[layerInfoRows.size()][0]);
    }

    //TODO float precision for smaller transfers?
    private Pair<List<Integer>, Map<String, List<Double>>> getLayerMeanMagnitudes(int layerIdx, TrainModuleUtils.GraphInfo gi,
                                                                                  List<Persistable> updates, boolean isMLN) {
        if(gi == null){
            return new Pair<>(Collections.emptyList(), Collections.emptyMap());
        }

        String layerName = gi.getVertexNames().get(layerIdx);
        if(isMLN){
            //Get the original name, for the index...
            layerName = gi.getOriginalVertexName().get(layerIdx);
        }
        String layerType = gi.getVertexTypes().get(layerIdx);
        if("input".equalsIgnoreCase(layerType)){        //TODO better checking - other vertices, etc
            return new Pair<>(Collections.emptyList(), Collections.emptyMap());
        }

        List<Integer> iterCounts = new ArrayList<>();
        Map<String, List<Double>> ratioValues = new HashMap<>();

        if (updates != null) {
            for (Persistable u : updates) {
                if (!(u instanceof StatsReport)) continue;
                StatsReport sp = (StatsReport) u;
                int iterCount = sp.getIterationCount();
                iterCounts.add(iterCount);

                //Info we want, for each parameter in this layer: mean magnitudes for parameters, updates AND the ratio of these
                Map<String, Double> paramMM = sp.getMeanMagnitudes(StatsType.Parameters);
                Map<String, Double> updateMM = sp.getMeanMagnitudes(StatsType.Updates);
                for (String s : paramMM.keySet()) {
                    String prefix = layerName + "_";
                    if (s.startsWith(prefix)) {
                        //Relevant parameter for this layer...
                        String layerParam = s.substring(prefix.length());
                        //TODO check and handle not collected case...
                        double pmm = paramMM.get(s);
                        double umm = updateMM.get(s);
                        double ratio = umm / pmm;
                        List<Double> list = ratioValues.get(layerParam);
                        if (list == null) {
                            list = new ArrayList<>();
                            ratioValues.put(layerParam, list);
                        }
                        list.add(ratio);
                    }
                }
            }
        }

        return new Pair<>(iterCounts, ratioValues);
    }


    private Triple<int[], float[], float[]> getLayerActivations(int index, TrainModuleUtils.GraphInfo gi, List<Persistable> updates, MultiLayerConfiguration conf, ComputationGraphConfiguration gConf) {
        if(gi == null){
            return new Triple<>(new int[0], new float[0], new float[0]);    //TODO reuse
        }

        String type = gi.getVertexTypes().get(index);    //Index may be for an input, for example
        if("input".equalsIgnoreCase(type)){
            return new Triple<>(new int[0], new float[0], new float[0]);    //TODO reuse
        }

        String layerName = gi.getOriginalVertexName().get(index);

        int size = (updates == null ? 0 : updates.size());
        int[] iterCounts = new int[size];
        float[] mean = new float[size];
        float[] stdev = new float[size];
        int used = 0;
        if (updates != null) {
            for (Persistable u : updates) {
                if (!(u instanceof StatsReport)) continue;
                StatsReport sp = (StatsReport) u;
                iterCounts[used] = sp.getIterationCount();

                Map<String, Double> means = sp.getMean(StatsType.Activations);
                Map<String, Double> stdevs = sp.getStdev(StatsType.Activations);

                //TODO PROPER VALIDATION ETC, ERROR HANDLING
                if (means != null && means.containsKey(layerName)) {
                    mean[used] = means.get(layerName).floatValue();
                    stdev[used] = stdevs.get(layerName).floatValue();
                    used++;
                }
            }
        }

        if (used != iterCounts.length) {
            iterCounts = Arrays.copyOf(iterCounts, used);
            mean = Arrays.copyOf(mean, used);
            stdev = Arrays.copyOf(stdev, used);
        }

        return new Triple<>(iterCounts, mean, stdev);
    }

    private Map<String, Object> getLayerLearningRates(int layerIdx, TrainModuleUtils.GraphInfo gi, List<Persistable> updates) {
        if(gi == null){
            return Collections.emptyMap();
        }
//        String layerName = gi.getVertexNames().get(layerIdx);
        String layerName = gi.getOriginalVertexName().get(layerIdx);

        int size = (updates == null ? 0 : updates.size());
        int[] iterCounts = new int[size];
        Map<String, float[]> byName = new HashMap<>();
        int used = 0;
        if (updates != null) {
            for (Persistable u : updates) {
                if (!(u instanceof StatsReport)) continue;
                StatsReport sp = (StatsReport) u;
                iterCounts[used] = sp.getIterationCount();

                //TODO PROPER VALIDATION ETC, ERROR HANDLING
                Map<String, Double> lrs = sp.getLearningRates();

                for (String p : lrs.keySet()) {
                    if (p.startsWith(layerName + "_")) {
                        String layerParamName = p.substring(Math.min(p.length(), layerName.length() + 1));
                        if (!byName.containsKey(layerParamName)) {
                            byName.put(layerParamName, new float[size]);
                        }
                        float[] lrThisParam = byName.get(layerParamName);
                        lrThisParam[used] = lrs.get(p).floatValue();
                    }
                }
                used++;
            }
        }

        List<String> paramNames = new ArrayList<>(byName.keySet());
        Collections.sort(paramNames);   //Sorted for consistency

        Map<String, Object> ret = new HashMap<>();
        ret.put("iterCounts", iterCounts);
        ret.put("paramNames", paramNames);
        ret.put("lrs", byName);

        return ret;
    }


    private static Map<String, Object> getHistograms(int layerIdx, TrainModuleUtils.GraphInfo gi, StatsType statsType, Persistable p) {
        if (p == null) return null;
        if (!(p instanceof StatsReport)) return null;
        StatsReport sr = (StatsReport) p;

        String layerName = gi.getOriginalVertexName().get(layerIdx);

        Map<String, Histogram> map = sr.getHistograms(statsType);

        List<String> paramNames = new ArrayList<>();

        Map<String, Object> ret = new HashMap<>();
        for (String s : map.keySet()) {
            if (s.startsWith(layerName)) {
                String paramName = s.substring(layerName.length() + 1);
                paramNames.add(paramName);
                Histogram h = map.get(s);
                Map<String, Object> thisHist = new HashMap<>();
                thisHist.put("min", h.getMin());
                thisHist.put("max", h.getMax());
                thisHist.put("bins", h.getNBins());
                thisHist.put("counts", h.getBinCounts());
                ret.put(paramName, thisHist);
            }
        }
        ret.put("paramNames", paramNames);

        return ret;
    }

    private static Map<String, Object> getMemory(List<Persistable> staticInfoAllWorkers, List<Persistable> updatesLastNMinutes, I18N i18n) {

        Map<String, Object> ret = new HashMap<>();

        //First: map workers to JVMs
        Set<String> jvmIDs = new HashSet<>();
        Map<String, String> workersToJvms = new HashMap<>();
        Map<String, Integer> workerNumDevices = new HashMap<>();
        Map<String, String[]> deviceNames = new HashMap<>();
        for (Persistable p : staticInfoAllWorkers) {
            //TODO validation/checks
            StatsInitializationReport init = (StatsInitializationReport) p;
            String jvmuid = init.getSwJvmUID();
            workersToJvms.put(p.getWorkerID(), jvmuid);
            jvmIDs.add(jvmuid);

            int nDevices = init.getHwNumDevices();
            workerNumDevices.put(p.getWorkerID(), nDevices);

            if (nDevices > 0) {
                String[] deviceNamesArr = init.getHwDeviceDescription();
                deviceNames.put(p.getWorkerID(), deviceNamesArr);
            }
        }

        List<String> jvmList = new ArrayList<>(jvmIDs);
        Collections.sort(jvmList);

        //For each unique JVM, collect memory info
        //Do this by selecting the first worker
        int count = 0;
        for (String jvm : jvmList) {
            List<String> workersForJvm = new ArrayList<>();
            for (String s : workersToJvms.keySet()) {
                if (workersToJvms.get(s).equals(jvm)) {
                    workersForJvm.add(s);
                }
            }
            Collections.sort(workersForJvm);
            String wid = workersForJvm.get(0);

            int numDevices = workerNumDevices.get(wid);

            Map<String, Object> jvmData = new HashMap<>();

            List<Long> timestamps = new ArrayList<>();
            List<Float> fracJvm = new ArrayList<>();
            List<Float> fracOffHeap = new ArrayList<>();
            long[] lastBytes = new long[2];
            long[] lastMaxBytes = new long[2];

            List<List<Float>> fracDeviceMem = null;
            if (numDevices > 0) {
                fracDeviceMem = new ArrayList<>(numDevices);
                for (int i = 0; i < numDevices; i++) {
                    fracDeviceMem.add(new ArrayList<>());
                }
            }

            for (Persistable p : updatesLastNMinutes) {
                //TODO single pass
                if (!p.getWorkerID().equals(wid)) continue;
                if (!(p instanceof StatsReport)) continue;

                StatsReport sp = (StatsReport) p;

                timestamps.add(sp.getTimeStamp());

                long jvmCurrentBytes = sp.getJvmCurrentBytes();
                long jvmMaxBytes = sp.getJvmMaxBytes();
                long ohCurrentBytes = sp.getOffHeapCurrentBytes();
                long ohMaxBytes = sp.getOffHeapMaxBytes();

                double jvmFrac = jvmCurrentBytes / ((double) jvmMaxBytes);
                double offheapFrac = ohCurrentBytes / ((double) ohMaxBytes);
                fracJvm.add((float) jvmFrac);
                fracOffHeap.add((float) offheapFrac);

                lastBytes[0] = jvmCurrentBytes;
                lastBytes[1] = ohCurrentBytes;

                lastMaxBytes[0] = jvmMaxBytes;
                lastMaxBytes[1] = ohMaxBytes;

                if (numDevices > 0) {
                    long[] devBytes = sp.getDeviceCurrentBytes();
                    long[] devMaxBytes = sp.getDeviceMaxBytes();
                    for (int i = 0; i < numDevices; i++) {
                        double frac = devBytes[i] / ((double) devMaxBytes[i]);
                        fracDeviceMem.get(i).add((float) frac);
                    }
                }

            }

            List<List<Float>> fracUtilized = new ArrayList<>();
            fracUtilized.add(fracJvm);
            fracUtilized.add(fracOffHeap);

            String[] seriesNames = new String[2 + numDevices];
            seriesNames[0] = i18n.getMessage("train.system.hwTable.jvmCurrent");
            seriesNames[1] = i18n.getMessage("train.system.hwTable.offHeapCurrent");
            boolean[] isDevice = new boolean[2 + numDevices];
            String[] devNames = deviceNames.get(wid);
            for (int i = 0; i < numDevices; i++) {
                seriesNames[2 + i] = devNames != null && devNames.length > i ? devNames[i] : "";
                fracUtilized.add(fracDeviceMem.get(i));
                isDevice[2 + i] = true;
            }

            jvmData.put("times", timestamps);
            jvmData.put("isDevice", isDevice);
            jvmData.put("seriesNames", seriesNames);
            jvmData.put("values", Arrays.asList(fracJvm, fracOffHeap));
            jvmData.put("currentBytes", lastBytes);
            jvmData.put("maxBytes", lastMaxBytes);
            ret.put(String.valueOf(count), jvmData);

            count++;
        }

        return ret;
    }

    private static Pair<Map<String, Object>, Map<String, Object>> getHardwareSoftwareInfo(List<Persistable> staticInfoAllWorkers, I18N i18n) {
        Map<String, Object> retHw = new HashMap<>();
        Map<String, Object> retSw = new HashMap<>();

        //First: map workers to JVMs
        Set<String> jvmIDs = new HashSet<>();
        Map<String, StatsInitializationReport> staticByJvm = new HashMap<>();
        for (Persistable p : staticInfoAllWorkers) {
            //TODO validation/checks
            StatsInitializationReport init = (StatsInitializationReport) p;
            String jvmuid = init.getSwJvmUID();
            jvmIDs.add(jvmuid);
            staticByJvm.put(jvmuid, init);
        }

        List<String> jvmList = new ArrayList<>(jvmIDs);
        Collections.sort(jvmList);

        //For each unique JVM, collect hardware info
        int count = 0;
        for (String jvm : jvmList) {
            StatsInitializationReport sr = staticByJvm.get(jvm);

            //---- Harware Info ----
            List<String[]> hwInfo = new ArrayList<>();
            int numDevices = sr.getHwNumDevices();
            String[] deviceDescription = sr.getHwDeviceDescription();
            long[] devTotalMem = sr.getHwDeviceTotalMemory();

            hwInfo.add(new String[]{i18n.getMessage("train.system.hwTable.jvmMax"), String.valueOf(sr.getHwJvmMaxMemory())});
            hwInfo.add(new String[]{i18n.getMessage("train.system.hwTable.offHeapMax"), String.valueOf(sr.getHwOffHeapMaxMemory())});
            hwInfo.add(new String[]{i18n.getMessage("train.system.hwTable.jvmProcs"), String.valueOf(sr.getHwJvmAvailableProcessors())});
            hwInfo.add(new String[]{i18n.getMessage("train.system.hwTable.computeDevices"), String.valueOf(numDevices)});
            for (int i = 0; i < numDevices; i++) {
                String label = i18n.getMessage("train.system.hardwareinfo.deviceName") + " (" + i + ")";
                String name = (deviceDescription == null || i >= deviceDescription.length ? String.valueOf(i) : deviceDescription[i]);
                hwInfo.add(new String[]{label, name});

                String memLabel = i18n.getMessage("train.system.hwTable.deviceMemory") + " (" + i + ")";
                String memBytes = (devTotalMem == null | i >= devTotalMem.length ? "-" : String.valueOf(devTotalMem[i]));
                hwInfo.add(new String[]{memLabel, memBytes});
            }

            retHw.put(String.valueOf(count), hwInfo);

            //---- Software Info -----

            String nd4jBackend = sr.getSwNd4jBackendClass();
            if(nd4jBackend != null && nd4jBackend.contains(".")){
                int idx = nd4jBackend.lastIndexOf('.');
                nd4jBackend = nd4jBackend.substring(idx+1);
                String temp;
                switch(nd4jBackend){
                    case "CpuNDArrayFactory":
                        temp = "CPU";
                        break;
                    case "JCublasNDArrayFactory":
                        temp = "CUDA";
                        break;
                    default:
                        temp = nd4jBackend;
                }
                nd4jBackend = temp;
            }

            String datatype = sr.getSwNd4jDataTypeName();
            if(datatype == null) datatype = "";
            else datatype = datatype.toLowerCase();

            List<String[]> swInfo = new ArrayList<>();
            swInfo.add(new String[]{i18n.getMessage("train.system.swTable.os"), sr.getSwOsName()});
            swInfo.add(new String[]{i18n.getMessage("train.system.swTable.hostname"), sr.getSwHostName()});
            swInfo.add(new String[]{i18n.getMessage("train.system.swTable.osArch"), sr.getSwArch()});
            swInfo.add(new String[]{i18n.getMessage("train.system.swTable.jvmName"), sr.getSwJvmName()});
            swInfo.add(new String[]{i18n.getMessage("train.system.swTable.jvmVersion"), sr.getSwJvmVersion()});
            swInfo.add(new String[]{i18n.getMessage("train.system.swTable.nd4jBackend"), nd4jBackend});
            swInfo.add(new String[]{i18n.getMessage("train.system.swTable.nd4jDataType"), datatype});

            retSw.put(String.valueOf(count), swInfo);

            count++;
        }

        return new Pair<>(retHw, retSw);
    }
}
