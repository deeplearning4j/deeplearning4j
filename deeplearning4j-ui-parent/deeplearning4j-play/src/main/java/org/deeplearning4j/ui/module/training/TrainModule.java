package org.deeplearning4j.ui.module.training;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.api.storage.StatsStorageListener;
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
import org.deeplearning4j.ui.stats.api.StatsInitializationReport;
import org.deeplearning4j.ui.stats.api.StatsReport;
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

    private static final String[][] EMPTY_TABLE = new String[2][0];

//    private static final JsonNode NO_DATA;
//    static {
//        Map<String,Object> noDataMap = new HashMap<>();
//        noDataMap.put("lastScore",0.0);
//        noDataMap.put("scores",Collections.EMPTY_LIST);
//        noDataMap.put("scoresIterCount",Collections.EMPTY_LIST);
//        noDataMap.put("performanceTable", new String[2][0]);
//        NO_DATA = Json.toJson(noDataMap);
//    }

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
        Route r5 = new Route("/train/help", HttpMethod.GET, FunctionType.Supplier, () -> ok(TrainingHelp.apply(I18NProvider.getInstance())));
        Route r6 = new Route("/train/currentSessionID", HttpMethod.GET, FunctionType.Supplier, () -> ok(currentSessionID == null ? "" : currentSessionID));

        return Arrays.asList(r, r2, r2a, r3, r3a, r4, r5, r6);
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

//        String[][] layerInfoTable = new String[][]{
//                {i18N.getMessage("train.model.layerinfotable.layerName"),layerID},
//                {i18N.getMessage("train.model.layerinfotable.layerType"),""},
//                {}
//        };

        List<String[]> layerInfoRows = new ArrayList<>();
        layerInfoRows.add(new String[]{i18N.getMessage("train.model.layerinfotable.layerName"), layerID});
        layerInfoRows.add(new String[]{i18N.getMessage("train.model.layerinfotable.layerType"), ""});

        // ----- Get static layer info -----
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
                        if(wi == WeightInit.DISTRIBUTION){
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

                    if(activationFn != null){
                        layerInfoRows.add(new String[]{i18N.getMessage("train.model.layerinfotable.layerActivationFn"), activationFn});
                    }
                }
                layerInfoRows.get(1)[1] = layerType;
            }
        }

        String[][] layerInfoTable = layerInfoRows.toArray(new String[layerInfoRows.size()][0]);

        result.put("layerInfo", layerInfoTable);


        return ok(Json.toJson(result));
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
}
