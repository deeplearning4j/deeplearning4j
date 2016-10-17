package org.deeplearning4j.ui.module.training;

import com.fasterxml.jackson.databind.JsonNode;
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

import java.util.*;

import static play.mvc.Results.ok;

/**
 * Created by Alex on 14/10/2016.
 */
@Slf4j
public class TrainModule implements UIModule {

    private Map<String, StatsStorage> knownSessionIDs = new LinkedHashMap<>();
    private String currentSessionID;


    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.singletonList(StatsListener.TYPE_ID);
    }

    @Override
    public List<Route> getRoutes() {
        Route r = new Route("/train", HttpMethod.GET, FunctionType.Supplier, () -> ok(Training.apply(I18NProvider.getInstance())));
        Route r2 = new Route("/train/overview", HttpMethod.GET, FunctionType.Supplier, () -> ok(TrainingOverview.apply(I18NProvider.getInstance())));
        Route r2a = new Route("/train/overview/data", HttpMethod.GET, FunctionType.Supplier, this::getOverviewData);
        Route r3 = new Route("/train/model", HttpMethod.GET, FunctionType.Supplier, () -> ok(TrainingModel.apply(I18NProvider.getInstance())));
        Route r3a = new Route("/train/model/data/:layerId", HttpMethod.GET, FunctionType.Function, this::getModelData);
        Route r4 = new Route("/train/system", HttpMethod.GET, FunctionType.Supplier, () -> ok(TrainingSystem.apply(I18NProvider.getInstance())));
        Route r4a = new Route("/train/system/data", HttpMethod.GET, FunctionType.Supplier, this::getSystemData);
        Route r5 = new Route("/train/help", HttpMethod.GET, FunctionType.Supplier, () -> ok(TrainingHelp.apply(I18NProvider.getInstance())));
        Route r6 = new Route("/train/sessions/current", HttpMethod.GET, FunctionType.Supplier, () -> ok(currentSessionID == null ? "" : currentSessionID));
        Route r6a = new Route("/train/sessions/all", HttpMethod.GET, FunctionType.Supplier, this::listSessions );
        Route r6b = new Route("/train/sessions/info", HttpMethod.GET, FunctionType.Supplier, this::sessionInfo );
        Route r6c = new Route("/train/sessions/set/:to", HttpMethod.GET, FunctionType.Function, this::setSession );

        return Arrays.asList(r, r2, r2a, r3, r3a, r4, r4a, r5, r6, r6a, r6b, r6c);
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

    private Result listSessions(){
        return Results.ok(Json.toJson(knownSessionIDs.keySet()));
    }

    private Result sessionInfo(){
        //Display, for each session: session ID, start time, number of workers, last update
        Map<String,Object> dataEachSession = new HashMap<>();
        for(Map.Entry<String,StatsStorage> entry : knownSessionIDs.entrySet()){
            Map<String,Object> dataThisSession = new HashMap<>();
            String sid = entry.getKey();
            StatsStorage ss = entry.getValue();
            List<String> workerIDs = ss.listWorkerIDsForSessionAndType(sid, StatsListener.TYPE_ID);
            int workerCount = (workerIDs == null ? 0 : workerIDs.size());
            List<Persistable> staticInfo = ss.getAllStaticInfos(sid, StatsListener.TYPE_ID);
            long initTime = Long.MAX_VALUE;
            if(staticInfo != null) {
                for (Persistable p : staticInfo) {
                    initTime = Math.min(p.getTimeStamp(), initTime);
                }
            }

            long lastUpdateTime = Long.MIN_VALUE;
            List<Persistable> lastUpdatesAllWorkers = ss.getLatestUpdateAllWorkers(sid, StatsListener.TYPE_ID);
            for(Persistable p : lastUpdatesAllWorkers){
                lastUpdateTime = Math.max(lastUpdateTime, p.getTimeStamp());
            }

            dataThisSession.put("numWorkers", workerCount);
            dataThisSession.put("initTime", initTime == Long.MAX_VALUE ? "" : initTime);
            dataThisSession.put("lastUpdate", lastUpdateTime == Long.MIN_VALUE ? "" : lastUpdateTime);

            //Model info: type, # layers, # params...
            if(staticInfo != null && staticInfo.size() > 0){
                StatsInitializationReport sr = (StatsInitializationReport)staticInfo.get(0);
                String modelClassName = sr.getModelClassName();
                if(modelClassName.endsWith("MultiLayerNetwork")){
                    modelClassName = "MultiLayerNetwork";
                } else if(modelClassName.endsWith("ComputationGraph")){
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

    private Result setSession(String newSessionID){
        if(knownSessionIDs.containsKey(newSessionID)){
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
            perfInfo[2][1] = String.valueOf(last.getTimeStamp());   //TODO FORMATTING
            perfInfo[3][1] = String.valueOf(last.getTotalMinibatches());
            perfInfo[4][1] = String.valueOf(last.getMinibatchesPerSecond());    //TODO FORMATTING
            perfInfo[5][1] = String.valueOf(last.getExamplesPerSecond());    //TODO FORMATTING
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


    private Result getModelData(String layerID) {
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


        // Get static layer info
        String[][] layerInfoTable = getLayerInfoTable(layerID, i18N, noData, ss, wid);

        result.put("layerInfo", layerInfoTable);

        //Get mean magnitudes line chart
        List<Persistable> updates = (noData ? null : ss.getAllUpdatesAfter(currentSessionID, StatsListener.TYPE_ID, wid, 0));
        Pair<List<Integer>, Map<String, List<Double>>> meanMagnitudes = getLayerMeanMagnitudes(layerID, updates);
        Map<String, Object> mmRatioMap = new HashMap<>();
        mmRatioMap.put("layerParamNames", meanMagnitudes.getSecond().keySet());
        mmRatioMap.put("iterCounts", meanMagnitudes.getFirst());
        for (Map.Entry<String, List<Double>> entry : meanMagnitudes.getSecond().entrySet()) {
            mmRatioMap.put(entry.getKey(), entry.getValue());
        }
        result.put("meanMagRatio", mmRatioMap);

        //Get activations line chart for layer
        Triple<int[], float[], float[]> activationsData = getLayerActivations(layerID, updates);
        Map<String, Object> activationMap = new HashMap<>();
        activationMap.put("iterCount", activationsData.getFirst());
        activationMap.put("mean", activationsData.getSecond());
        activationMap.put("stdev", activationsData.getThird());
        result.put("activations", activationMap);

        //Get learning rate vs. time chart for layer
        Map<String,Object> lrs = getLayerLearningRates(layerID, updates);
        result.put("learningRates", lrs);

        //Parameters histogram data
        Persistable lastUpdate = (updates != null && updates.size() > 0 ? updates.get(updates.size() - 1) : null);
        Map<String, Object> paramHistograms = getHistograms(layerID, StatsType.Parameters, lastUpdate);
        result.put("paramHist", paramHistograms);

        //Updates histogram data
        Map<String, Object> updateHistograms = getHistograms(layerID, StatsType.Updates, lastUpdate);
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

    private String[][] getLayerInfoTable(String layerID, I18N i18N, boolean noData, StatsStorage ss, String wid) {
        List<String[]> layerInfoRows = new ArrayList<>();
        layerInfoRows.add(new String[]{i18N.getMessage("train.model.layerinfotable.layerName"), layerID});
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
                    int layerIdx = Integer.parseInt(layerID);
                    if (layerIdx >= 0) {
                        nnc = conf.getConf(layerIdx);
                        layer = nnc.getLayer();
                    }
                } else if (modelClass.endsWith("ComputationGraph")) {
                    ComputationGraphConfiguration conf = ComputationGraphConfiguration.fromJson(configJson);
                    Map<String, GraphVertex> vertices = conf.getVertices();
                    if (vertices.containsKey(layerID) && vertices.get(layerID) instanceof LayerVertex) {
                        LayerVertex lv = (LayerVertex) vertices.get(layerID);
                        nnc = lv.getLayerConf();
                        layer = nnc.getLayer();
                    }
                }
                layerType = getLayerType(layer);
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
    private Pair<List<Integer>, Map<String, List<Double>>> getLayerMeanMagnitudes(String layerID, List<Persistable> updates) {

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
                    String prefix = layerID + "_";
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


    private Triple<int[], float[], float[]> getLayerActivations(String paramName, List<Persistable> updates) {

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
                if (means != null && means.containsKey(paramName)) {
                    mean[used] = means.get(paramName).floatValue();
                    stdev[used] = stdevs.get(paramName).floatValue();
                } else {
                    mean[used] = 0.0f;
                    stdev[used] = 1.0f;
                }

                used++;
            }
        }

        if (used != iterCounts.length) {
            iterCounts = Arrays.copyOf(iterCounts, used);
            mean = Arrays.copyOf(mean, used);
            stdev = Arrays.copyOf(stdev, used);
        }

        return new Triple<>(iterCounts, mean, stdev);
    }

    private Map<String,Object> getLayerLearningRates(String paramName, List<Persistable> updates) {

        int size = (updates == null ? 0 : updates.size());
        int[] iterCounts = new int[size];
        Map<String,float[]> byName = new HashMap<>();
        int used = 0;
        if (updates != null) {
            for (Persistable u : updates) {
                if (!(u instanceof StatsReport)) continue;
                StatsReport sp = (StatsReport) u;
                iterCounts[used] = sp.getIterationCount();

                //TODO PROPER VALIDATION ETC, ERROR HANDLING
                Map<String,Double> lrs = sp.getLearningRates();

                for(String p : lrs.keySet()){
                    if(p.startsWith(paramName + "_")){
                        String layerParamName = p.substring(Math.min(p.length(), paramName.length() + 1));
                        if(!byName.containsKey(layerParamName)){
                            byName.put(layerParamName,new float[size]);
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

        Map<String,Object> ret = new HashMap<>();
        ret.put("iterCounts", iterCounts);
        ret.put("paramNames", paramNames);
        ret.put("lrs", byName);

        return ret;
    }


    private static Map<String, Object> getHistograms(String layerName, StatsType statsType, Persistable p) {
        if (p == null) return null;
        if (!(p instanceof StatsReport)) return null;
        StatsReport sr = (StatsReport) p;


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

                lastMaxBytes[0] = jvmCurrentBytes;
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
            seriesNames[0] = i18n.getMessage("train.system.memory.onHeapName");
            seriesNames[1] = i18n.getMessage("train.system.memory.offHeapName");
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

            hwInfo.add(new String[]{i18n.getMessage("train.system.hardwareinfo.jvmMaxMem"), String.valueOf(sr.getHwJvmMaxMemory())});
            hwInfo.add(new String[]{i18n.getMessage("train.system.hardwareinfo.jvmMaxMem"), String.valueOf(sr.getHwOffHeapMaxMemory())});
            hwInfo.add(new String[]{i18n.getMessage("train.system.hardwareinfo.jvmprocs"), String.valueOf(sr.getHwJvmAvailableProcessors())});
            hwInfo.add(new String[]{i18n.getMessage("train.system.hardwareinfo.numDevices"), String.valueOf(numDevices)});
            for (int i = 0; i < numDevices; i++) {
                String label = i18n.getMessage("train.system.hardwareinfo.deviceName") + " (" + i + ")";
                String name = (deviceDescription == null || i >= deviceDescription.length ? String.valueOf(i) : deviceDescription[i]);
                hwInfo.add(new String[]{label, name});

                String memLabel = i18n.getMessage("train.system.hardwareinfo.deviceMemory") + " (" + i + ")";
                String memBytes = (devTotalMem == null | i >= devTotalMem.length ? "-" : String.valueOf(devTotalMem[i]));
                hwInfo.add(new String[]{memLabel, memBytes});
            }

            retHw.put(String.valueOf(count), hwInfo);

            //---- Software Info -----
            List<String[]> swInfo = new ArrayList<>();
            swInfo.add(new String[]{i18n.getMessage("train.system.softwareinfo.os"), sr.getSwOsName()});
            swInfo.add(new String[]{i18n.getMessage("train.system.softwareinfo.hostname"), sr.getSwHostName()});
            swInfo.add(new String[]{i18n.getMessage("train.system.softwareinfo.architecture"), sr.getSwArch()});
            swInfo.add(new String[]{i18n.getMessage("train.system.softwareinfo.jvmName"), sr.getSwJvmName()});
            swInfo.add(new String[]{i18n.getMessage("train.system.softwareinfo.jvmVersion"), sr.getSwJvmVersion()});
            swInfo.add(new String[]{i18n.getMessage("train.system.softwareinfo.nd4jBackend"), sr.getSwNd4jBackendClass()});     //TODO proper formatting
            swInfo.add(new String[]{i18n.getMessage("train.system.softwareinfo.nd4jDataType"), sr.getSwNd4jDataTypeName()});

            retSw.put(String.valueOf(count), swInfo);

            count++;
        }

        return new Pair<>(retHw, retSw);
    }
}
