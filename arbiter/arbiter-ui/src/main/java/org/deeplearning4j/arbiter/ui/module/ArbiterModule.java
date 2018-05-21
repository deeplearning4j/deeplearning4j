package org.deeplearning4j.arbiter.ui.module;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.api.storage.StatsStorageListener;
import org.deeplearning4j.arbiter.BaseNetworkSpace;
import org.deeplearning4j.arbiter.layers.LayerSpace;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.runner.CandidateStatus;
import org.deeplearning4j.arbiter.ui.UpdateStatus;
import org.deeplearning4j.arbiter.ui.data.GlobalConfigPersistable;
import org.deeplearning4j.arbiter.ui.data.ModelInfoPersistable;
import org.deeplearning4j.arbiter.ui.misc.UIUtils;
import org.deeplearning4j.arbiter.ui.views.html.ArbiterUI;
import org.deeplearning4j.arbiter.util.ObjectUtils;
import org.deeplearning4j.ui.api.Component;
import org.deeplearning4j.ui.api.*;
import org.deeplearning4j.ui.components.chart.ChartLine;
import org.deeplearning4j.ui.components.chart.ChartScatter;
import org.deeplearning4j.ui.components.chart.style.StyleChart;
import org.deeplearning4j.ui.components.component.ComponentDiv;
import org.deeplearning4j.ui.components.component.style.StyleDiv;
import org.deeplearning4j.ui.components.table.ComponentTable;
import org.deeplearning4j.ui.components.table.style.StyleTable;
import org.deeplearning4j.ui.components.text.ComponentText;
import org.deeplearning4j.ui.components.text.style.StyleText;
import org.deeplearning4j.ui.i18n.I18NResource;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;
import org.nd4j.linalg.primitives.Pair;
import play.libs.Json;
import play.mvc.Result;
import play.mvc.Results;

import java.awt.*;
import java.text.DecimalFormat;
import java.util.*;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.deeplearning4j.arbiter.ui.misc.JsonMapper.asJson;
import static play.mvc.Results.ok;

/**
 * A Deeplearning4j {@link UIModule}, for integration with DL4J's user interface
 *
 * @author Alex Black
 */
@Slf4j
public class ArbiterModule implements UIModule {

    private static final DecimalFormat DECIMAL_FORMAT_2DP = new DecimalFormat("#.00");
    private static final DateTimeFormatter TIME_FORMATTER = DateTimeFormat.forPattern("YYYY-MM-dd HH:mm ZZ");
    public static final String ARBITER_UI_TYPE_ID = "ArbiterUI";

    private static final String JSON = "application/json";

    private AtomicBoolean loggedArbiterAddress = new AtomicBoolean(false);
    private Map<String, StatsStorage> knownSessionIDs = Collections.synchronizedMap(new LinkedHashMap<>());
    private String currentSessionID;

    private Map<String, Long> lastUpdateForSession = Collections.synchronizedMap(new HashMap<>());


    //Styles for UI:
    private static final StyleTable STYLE_TABLE = new StyleTable.Builder()
            .width(100, LengthUnit.Percent)
            .backgroundColor(Color.WHITE)
            .borderWidth(1)
            .columnWidths(LengthUnit.Percent, 30, 70)
            .build();

    private static final StyleTable STYLE_TABLE3_25_25_50 = new StyleTable.Builder()
            .width(100, LengthUnit.Percent)
            .backgroundColor(Color.WHITE)
            .borderWidth(1)
            .columnWidths(LengthUnit.Percent, 25, 25, 50)
            .build();

    private static final StyleDiv STYLE_DIV_WIDTH_100_PC = new StyleDiv.Builder()
            .width(100, LengthUnit.Percent)
            .build();

    private static final ComponentDiv DIV_SPACER_20PX = new ComponentDiv(new StyleDiv.Builder()
            .width(100,LengthUnit.Percent)
            .height(20, LengthUnit.Px).build());

    private static final ComponentDiv DIV_SPACER_60PX = new ComponentDiv(new StyleDiv.Builder()
            .width(100,LengthUnit.Percent)
            .height(60, LengthUnit.Px).build());

    private static final StyleChart STYLE_CHART_560_320 = new StyleChart.Builder()
            .width(560, LengthUnit.Px)
            .height(320, LengthUnit.Px)
            .build();

    private static final StyleChart STYLE_CHART_800_400 = new StyleChart.Builder()
            .width(800, LengthUnit.Px)
            .height(400, LengthUnit.Px)
            .build();


    private StyleText STYLE_TEXT_SZ12 = new StyleText.Builder()
            .fontSize(12)
            .build();

    //Set whitespacePre(true) to avoid losing new lines, tabs, multiple spaces etc
    private StyleText STYLE_TEXT_SZ10_WHITESPACE_PRE = new StyleText.Builder()
            .fontSize(10)
            .whitespacePre(true)
            .build();


    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.singletonList(ARBITER_UI_TYPE_ID);
    }

    @Override
    public List<Route> getRoutes() {
        Route r1 = new Route("/arbiter", HttpMethod.GET, FunctionType.Supplier, () -> Results.ok(ArbiterUI.apply()));
        Route r3 = new Route("/arbiter/lastUpdate", HttpMethod.GET, FunctionType.Supplier, this::getLastUpdateTime);
        Route r4 = new Route("/arbiter/lastUpdate/:ids", HttpMethod.GET, FunctionType.Function, this::getModelLastUpdateTimes);
        Route r5 = new Route("/arbiter/candidateInfo/:id", HttpMethod.GET, FunctionType.Function, this::getCandidateInfo);
        Route r6 = new Route("/arbiter/config", HttpMethod.GET, FunctionType.Supplier, this::getOptimizationConfig);
        Route r7 = new Route("/arbiter/results", HttpMethod.GET, FunctionType.Supplier, this::getSummaryResults);
        Route r8 = new Route("/arbiter/summary", HttpMethod.GET, FunctionType.Supplier, this::getSummaryStatus);

        Route r9a = new Route("/arbiter/sessions/all", HttpMethod.GET, FunctionType.Supplier, this::listSessions);
        Route r9b = new Route("/arbiter/sessions/current", HttpMethod.GET, FunctionType.Supplier, this::currentSession);
        Route r9c = new Route("/arbiter/sessions/set/:to", HttpMethod.GET, FunctionType.Function, this::setSession);

        return Arrays.asList(r1, r3, r4, r5, r6, r7, r8, r9a, r9b, r9c);
    }

    @Override
    public void reportStorageEvents(Collection<StatsStorageEvent> events) {
        boolean attachedArbiter = false;
        for (StatsStorageEvent sse : events) {
            if (ARBITER_UI_TYPE_ID.equals(sse.getTypeID())) {
                if (sse.getEventType() == StatsStorageListener.EventType.PostStaticInfo) {
                    knownSessionIDs.put(sse.getSessionID(), sse.getStatsStorage());
                }

                Long lastUpdate = lastUpdateForSession.get(sse.getSessionID());
                if (lastUpdate == null) {
                    lastUpdateForSession.put(sse.getSessionID(), sse.getTimestamp());
                } else if (sse.getTimestamp() > lastUpdate) {
                    lastUpdateForSession.put(sse.getSessionID(), sse.getTimestamp()); //Should be thread safe - read only elsewhere
                }
                attachedArbiter = true;
            }
        }

        if(currentSessionID == null){
            getDefaultSession();
        }

        if(attachedArbiter && !loggedArbiterAddress.getAndSet(true)){
            String address = UIServer.getInstance().getAddress();
            address += "/arbiter";
            log.info("DL4J Arbiter Hyperparameter Optimization UI: {}", address);
        }
    }

    @Override
    public synchronized void onAttach(StatsStorage statsStorage) {
        for (String sessionID : statsStorage.listSessionIDs()) {
            for (String typeID : statsStorage.listTypeIDsForSession(sessionID)) {
                if (!ARBITER_UI_TYPE_ID.equals(typeID))
                    continue;
                knownSessionIDs.put(sessionID, statsStorage);
            }
        }

        if (currentSessionID == null)
            getDefaultSession();
    }

    private Result currentSession() {
        String sid = currentSessionID == null ? "" : currentSessionID;
        return ok(asJson(sid)).as(JSON);
    }

    private Result listSessions() {
        return Results.ok(asJson(knownSessionIDs.keySet())).as(JSON);
    }

    private Result setSession(String newSessionID) {
        log.debug("Arbiter UI: Set to session {}", newSessionID);

        if (knownSessionIDs.containsKey(newSessionID)) {
            currentSessionID = newSessionID;
            return ok();
        } else {
            return Results.badRequest("Unknown session ID: " + newSessionID);
        }
    }

    private void getDefaultSession() {
        if (currentSessionID != null)
            return;

        long mostRecentTime = Long.MIN_VALUE;
        String sessionID = null;
        for (Map.Entry<String, StatsStorage> entry : knownSessionIDs.entrySet()) {
            List<Persistable> staticInfos = entry.getValue().getAllStaticInfos(entry.getKey(), ARBITER_UI_TYPE_ID);
            if (staticInfos == null || staticInfos.size() == 0)
                continue;
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

    @Override
    public void onDetach(StatsStorage statsStorage) {
        for (String s : knownSessionIDs.keySet()) {
            if (knownSessionIDs.get(s) == statsStorage) {
                knownSessionIDs.remove(s);
            }
        }
    }

    @Override
    public List<I18NResource> getInternationalizationResources() {
        return Collections.emptyList();
    }

    /**
     * @return Last update time for the page
     */
    private Result getLastUpdateTime(){
        //TODO - this forces updates on every request... which is fine, just inefficient
        long t = System.currentTimeMillis();
        UpdateStatus us = new UpdateStatus(t, t, t);

        return ok(Json.toJson(us));
    }

    /**
     * Get the last update time for the specified model IDs
     * @param modelIDs Model IDs to get the update time for
     */
    private Result getModelLastUpdateTimes(String modelIDs){

        if(currentSessionID == null){
            return ok();
        }

        StatsStorage ss = knownSessionIDs.get(currentSessionID);
        if(ss == null){
            log.debug("getModelLastUpdateTimes(): Session ID is unknown: {}", currentSessionID);
            return ok("-1");
        }

        String[] split = modelIDs.split(",");

        long[] lastUpdateTimes = new long[split.length];
        for( int i=0; i<split.length; i++ ){
            String s = split[i];
            Persistable p = ss.getLatestUpdate(currentSessionID, ARBITER_UI_TYPE_ID, s);
            if(p != null){
                lastUpdateTimes[i] = p.getTimeStamp();
            }
        }

        return ok(Json.toJson(lastUpdateTimes));
    }

    /**
     * Get the info for a specific candidate - last section in the UI
     *
     * @param candidateId ID for the candidate
     * @return Content/info for the candidate
     */
    private Result getCandidateInfo(String candidateId){

        StatsStorage ss = knownSessionIDs.get(currentSessionID);
        if(ss == null){
            log.debug("getModelLastUpdateTimes(): Session ID is unknown: {}", currentSessionID);
            return ok();
        }

        GlobalConfigPersistable gcp = (GlobalConfigPersistable)ss.getStaticInfo(currentSessionID, ARBITER_UI_TYPE_ID, GlobalConfigPersistable.GLOBAL_WORKER_ID);;
        OptimizationConfiguration oc = gcp.getOptimizationConfiguration();

        Persistable p = ss.getLatestUpdate(currentSessionID, ARBITER_UI_TYPE_ID, candidateId);
        if(p == null){
            String title = "No results found for model " + candidateId + ".";
            ComponentText ct = new ComponentText.Builder(title,STYLE_TEXT_SZ12).build();
            return ok(asJson(ct)).as(JSON);
        }

        ModelInfoPersistable mip = (ModelInfoPersistable)p;

        //First: static info
        // Hyperparameter configuration/settings
        // Number of parameters
        // Maybe memory info in the future?

        //Second: dynamic info
        //Runtime
        // Performance stats (total minibatches, total time,
        // Score vs. time

        List<Component> components = new ArrayList<>();

        //First table: mix of static + dynamic in a table
        long runtimeDurationMs = mip.getLastUpdateTime() - mip.getTimeStamp();
        double avgMinibatchesPerSec = mip.getTotalNumUpdates() / (runtimeDurationMs/1000.0);
        String avgMinibatchesPerSecStr = DECIMAL_FORMAT_2DP.format(avgMinibatchesPerSec);
        String runtimeStr = UIUtils.formatDuration(runtimeDurationMs);

        if(mip.getStatus() == CandidateStatus.Failed){
            runtimeStr = "";
            avgMinibatchesPerSecStr = "";
        }

        String[][] table = new String[][]{
                {"Model Index", String.valueOf(mip.getModelIdx())},
                {"Status", mip.getStatus().toString()},
                {"Model Score", mip.getScore() == null ? "" : String.valueOf(mip.getScore())},
                {"Created", TIME_FORMATTER.print(mip.getTimeStamp())},
                {"Runtime", runtimeStr},
                {"Total Number of Model Updates", String.valueOf(mip.getTotalNumUpdates())},
                {"Average # Updates / Sec", avgMinibatchesPerSecStr},
                {"Number of Parameters", String.valueOf(mip.getNumParameters())},
                {"Number of Layers", String.valueOf(mip.getNumLayers())}
        };

        ComponentTable cTable = new ComponentTable.Builder(STYLE_TABLE)
                .content(table)
                .header("Model Information", "")
                .build();
        components.add(cTable);


        //Second: parameter space values, in multiple tables
        double[] paramSpaceValues = mip.getParamSpaceValues();
        if(paramSpaceValues != null){
            BaseNetworkSpace bns = (BaseNetworkSpace)oc.getCandidateGenerator().getParameterSpace();
            Map<String,ParameterSpace> m = bns.getNestedSpaces();

            String[][] hSpaceTable = new String[m.size()][3];
            int i=0;
            for(Map.Entry<String,ParameterSpace> e : m.entrySet()){
                hSpaceTable[i][0] = e.getKey();
                Object currCandidateValue = e.getValue().getValue(paramSpaceValues);
                hSpaceTable[i][1] = ObjectUtils.valueToString(currCandidateValue);
                hSpaceTable[i][2] = e.getValue().toString();
                i++;
            }

            String[] hSpaceTableHeader = new String[]{"Hyperparameter", "Model Value", "Hyperparameter Space"};

            ComponentTable ct2 = new ComponentTable.Builder(STYLE_TABLE3_25_25_50)
                    .content(hSpaceTable)
                    .header(hSpaceTableHeader)
                    .build();


            String title = "Global Network Configuration";
            components.add(DIV_SPACER_20PX);
            components.add(new ComponentText.Builder(title, STYLE_TEXT_SZ12).build());
            components.add(ct2);

            List<BaseNetworkSpace.LayerConf> layerConfs = bns.getLayerSpaces();

            for(BaseNetworkSpace.LayerConf l : layerConfs){
                LayerSpace<?> ls = l.getLayerSpace();
                Map<String,ParameterSpace> lpsm = ls.getNestedSpaces();

                String[][] t = new String[lpsm.size()][3];
                i=0;
                for(Map.Entry<String,ParameterSpace> e : lpsm.entrySet()){
                    t[i][0] = e.getKey();
                    Object currCandidateValue = e.getValue().getValue(paramSpaceValues);
                    t[i][1] = ObjectUtils.valueToString(currCandidateValue);
                    t[i][2] = e.getValue().toString();
                    i++;
                }

                ComponentTable ct3 = new ComponentTable.Builder(STYLE_TABLE3_25_25_50)
                        .content(t)
                        .header(hSpaceTableHeader)
                        .build();

                title = "Layer Space: " + ls.getClass().getSimpleName() + ", Name: " + l.getLayerName();

                components.add(DIV_SPACER_20PX);
                components.add(new ComponentText.Builder(title, STYLE_TEXT_SZ12).build());
                components.add(ct3);
            }
        }


        //Third: Score vs. time chart
        int[] iters = mip.getIter();
        float[] scores = mip.getScoreVsIter();

        if(iters != null) {
            double[] si = new double[iters.length];
            double[] scoresD = new double[iters.length];

            double minScore = Double.MAX_VALUE;
            double maxScore = -Double.MAX_VALUE;
            for( int i=0; i<iters.length; i++ ){
                si[i] = iters[i];
                scoresD[i] = scores[i];
                minScore = Math.min(minScore, scoresD[i]);
                maxScore = Math.max(maxScore, scoresD[i]);
            }

            double[] chartMinMax = UIUtils.graphNiceRange(maxScore, minScore, 5);

            ChartLine cl = new ChartLine.Builder("Model Score vs. Iteration", STYLE_CHART_800_400)
                    .addSeries("Score", si, scoresD )
                    .setYMin(chartMinMax[0])
                    .setYMax(chartMinMax[1])
                    .build();

            components.add(DIV_SPACER_60PX);
            components.add(cl);
        }


        //Post full network configuration JSON, if available:
        String modelJson = mip.getModelConfigJson();
        if(modelJson != null){
            components.add(DIV_SPACER_60PX);
            components.add(new ComponentDiv(STYLE_DIV_WIDTH_100_PC, new ComponentText("Model Configuration", STYLE_TEXT_SZ12)));
            ComponentText jsonText = new ComponentText(modelJson, STYLE_TEXT_SZ10_WHITESPACE_PRE);
            ComponentDiv cd = new ComponentDiv(STYLE_DIV_WIDTH_100_PC, jsonText);
            components.add(cd);
        }


        //Post exception stack trace, if necessary:
        if( mip.getExceptionStackTrace() != null ){
            components.add(DIV_SPACER_60PX);
            components.add(new ComponentDiv(STYLE_DIV_WIDTH_100_PC, new ComponentText("Model Exception - Stack Trace", STYLE_TEXT_SZ12)));
            ComponentText exText = new ComponentText(mip.getExceptionStackTrace(), STYLE_TEXT_SZ10_WHITESPACE_PRE);
            ComponentDiv cd = new ComponentDiv(STYLE_DIV_WIDTH_100_PC, exText);
            components.add(cd);
        }

        ComponentDiv cd = new ComponentDiv(STYLE_DIV_WIDTH_100_PC, components);

        return ok(asJson(cd)).as(JSON);
    }

    /**
     * Get the optimization configuration - second section in the page
     */
    private Result getOptimizationConfig(){

        StatsStorage ss = knownSessionIDs.get(currentSessionID);
        if(ss == null){
            log.debug("getOptimizationConfig(): Session ID is unknown: {}", currentSessionID);
            return ok();
        }

        Persistable p = ss.getStaticInfo(currentSessionID, ARBITER_UI_TYPE_ID, GlobalConfigPersistable.GLOBAL_WORKER_ID);

        if(p == null){
            log.info("No static info");
            return ok();
        }

        List<Component> components = new ArrayList<>();

        GlobalConfigPersistable gcp = (GlobalConfigPersistable)p;
        OptimizationConfiguration oc = gcp.getOptimizationConfiguration();

        //Report optimization settings/configuration.
        String[] tableHeader = {"Configuration", "Value"};
        String[][] table = new String[][]{
                {"Candidate Generator", oc.getCandidateGenerator().getClass().getSimpleName()},
                {"Data Provider", oc.getDataProvider().toString()},
                {"Score Function", oc.getScoreFunction().toString()},
                {"Result Saver", oc.getResultSaver().toString()},
        };

        ComponentTable ct = new ComponentTable.Builder(STYLE_TABLE)
                .content(table)
                .header(tableHeader)
                .build();
        components.add(ct);


        String title = "Global Network Configuration";
        components.add(DIV_SPACER_20PX);
        components.add(new ComponentText.Builder(title, STYLE_TEXT_SZ12).build());
        BaseNetworkSpace<?> ps = (BaseNetworkSpace)oc.getCandidateGenerator().getParameterSpace();
        Map<String,ParameterSpace> m = ps.getNestedSpaces();

        String[][] hSpaceTable = new String[m.size()][2];
        int i=0;
        for(Map.Entry<String,ParameterSpace> e : m.entrySet()){
            hSpaceTable[i][0] = e.getKey();
            hSpaceTable[i][1] = e.getValue().toString();
            i++;
        }

        components.add(DIV_SPACER_20PX);
        String[] hSpaceTableHeader = new String[]{"Hyperparameter", "Hyperparameter Configuration"};

        ComponentTable ct2 = new ComponentTable.Builder(STYLE_TABLE)
                .content(hSpaceTable)
                .header(hSpaceTableHeader)
                .build();
        components.add(ct2);

        //Configuration for each layer:
        List<BaseNetworkSpace.LayerConf> layerConfs = ps.getLayerSpaces();
        for(BaseNetworkSpace.LayerConf l : layerConfs){
            LayerSpace<?> ls = l.getLayerSpace();
            Map<String,ParameterSpace> lpsm = ls.getNestedSpaces();

            String[][] t = new String[lpsm.size()][2];
            i=0;
            for(Map.Entry<String,ParameterSpace> e : lpsm.entrySet()){
                t[i][0] = e.getKey();
                t[i][1] = e.getValue().toString();
                i++;
            }

            ComponentTable ct3 = new ComponentTable.Builder(STYLE_TABLE)
                    .content(t)
                    .header(hSpaceTableHeader)
                    .build();

            title = "Layer Space: " + ls.getClass().getSimpleName() + ", Name: " + l.getLayerName();

            components.add(DIV_SPACER_20PX);
            components.add(new ComponentText.Builder(title, STYLE_TEXT_SZ12).build());
            components.add(ct3);
        }

        ComponentDiv cd = new ComponentDiv(STYLE_DIV_WIDTH_100_PC, components);

        return ok(asJson(cd)).as(JSON);
    }

    private Result getSummaryResults(){
        StatsStorage ss = knownSessionIDs.get(currentSessionID);
        if(ss == null){
            log.debug("getSummaryResults(): Session ID is unknown: {}", currentSessionID);
            return ok();
        }

        List<Persistable> allModelInfoTemp = new ArrayList<>(ss.getLatestUpdateAllWorkers(currentSessionID, ARBITER_UI_TYPE_ID));
        List<String[]> table = new ArrayList<>();
        for(Persistable per : allModelInfoTemp){
            ModelInfoPersistable mip = (ModelInfoPersistable)per;
            String score = (mip.getScore() == null ? "" : mip.getScore().toString());
            table.add(new String[]{mip.getModelIdx().toString(), score, mip.getStatus().toString()});
        }

        return ok(asJson(table)).as(JSON);
    }

    /**
     * Get summary status information: first section in the page
     */
    private Result getSummaryStatus(){
        StatsStorage ss = knownSessionIDs.get(currentSessionID);
        if(ss == null){
            log.debug("getOptimizationConfig(): Session ID is unknown: {}", currentSessionID);
            return ok();
        }

        Persistable p = ss.getStaticInfo(currentSessionID, ARBITER_UI_TYPE_ID, GlobalConfigPersistable.GLOBAL_WORKER_ID);

        if(p == null){
            log.info("No static info");
            return ok();
        }

        GlobalConfigPersistable gcp = (GlobalConfigPersistable)p;
        OptimizationConfiguration oc = gcp.getOptimizationConfiguration();
        long execStartTime = oc.getExecutionStartTime();



        //Charts:
        //Best model score vs. time
        //All candidate scores (scatter plot vs. time)

        //How to get this? query all model infos...

        List<Persistable> allModelInfoTemp = new ArrayList<>(ss.getLatestUpdateAllWorkers(currentSessionID, ARBITER_UI_TYPE_ID));
        List<ModelInfoPersistable> allModelInfo = new ArrayList<>();
        for(Persistable per : allModelInfoTemp){
            ModelInfoPersistable mip = (ModelInfoPersistable)per;
            if(mip.getStatus() == CandidateStatus.Complete && mip.getScore() != null && Double.isFinite(mip.getScore())){
                allModelInfo.add(mip);
            }
        }

        allModelInfo.sort(Comparator.comparingLong(Persistable::getTimeStamp));

        Pair<List<Component>, ModelInfoPersistable> chartsAndBest = getSummaryChartsAndBest(allModelInfo, oc.getScoreFunction().minimize(), execStartTime );

        //First: table - number completed, queued, running, failed, total
        //Best model index, score, and time
        //Total runtime
        //Termination conditions
        List<Component> components = new ArrayList<>();



        List<TerminationCondition> tcs = oc.getTerminationConditions();

        //TODO: I18N

        //TODO don't use currentTimeMillis due to stored data??
        long bestTime;
        Double bestScore = null;
        String bestModelString = null;
        if(chartsAndBest.getSecond() != null){
            bestTime = chartsAndBest.getSecond().getTimeStamp();
            bestScore = chartsAndBest.getSecond().getScore();
            String sinceBest = UIUtils.formatDuration(System.currentTimeMillis() - bestTime);

            bestModelString = "Model " + chartsAndBest.getSecond().getModelIdx() + ", Found at " +
            TIME_FORMATTER.print(bestTime) + " (" + sinceBest + " ago)";
        }

        String execStartTimeStr = "";
        String execTotalRuntimeStr = "";
        if(execStartTime > 0){
            execStartTimeStr = TIME_FORMATTER.print(execStartTime);
            execTotalRuntimeStr = UIUtils.formatDuration(System.currentTimeMillis() - execStartTime);
        }


        String[][] table = new String[][]{
                {"Models Completed", String.valueOf(gcp.getCandidatesCompleted())},
                {"Models Queued/Running", String.valueOf(gcp.getCandidatesQueued())},
                {"Models Failed", String.valueOf(gcp.getCandidatesFailed())},
                {"Models Total", String.valueOf(gcp.getCandidatesTotal())},
                {"Best Score", (bestScore != null ? String.valueOf(bestScore) : "")},
                {"Best Scoring Model", bestModelString != null ? bestModelString : ""},
                {"Optimization Runner", gcp.getOptimizationRunner()},
                {"Execution Start Time", execStartTimeStr},
                {"Total Runtime", execTotalRuntimeStr}
        };



        ComponentTable ct = new ComponentTable.Builder(STYLE_TABLE)
                .content(table)
                .header("Status", "")
                .build();

        components.add(ct);

        String[][] tcTable = new String[tcs.size()][2];
        for( int i=0; i<tcs.size(); i++ ){
            tcTable[i][0] = "Termination Condition " + i;
            tcTable[i][1] = tcs.get(i).toString();
        }

        components.add(DIV_SPACER_20PX);

        ComponentTable ct2 = new ComponentTable.Builder(STYLE_TABLE)
                .content(tcTable)
                .header("Termination Condition", "")
                .build();

        components.add(ct2);

        components.addAll(chartsAndBest.getFirst());


        ComponentDiv cd = new ComponentDiv(STYLE_DIV_WIDTH_100_PC, components);

        return ok(asJson(cd)).as(JSON);
    }


    private Pair<List<Component>,ModelInfoPersistable> getSummaryChartsAndBest(List<ModelInfoPersistable> allModelInfo,
                                                                               boolean minimize, long execStartTime){
        List<Double> bestX = new ArrayList<>();
        List<Double> bestY = new ArrayList<>();

        double[] allX = new double[allModelInfo.size()];
        double[] allY = new double[allModelInfo.size()];

        double bestScore = (minimize ? Double.MAX_VALUE : -Double.MAX_VALUE);
        double worstScore = (minimize ? -Double.MAX_VALUE : Double.MAX_VALUE);
        double lastTime = -1L;
        ModelInfoPersistable bestModel = null;
        for(int i=0; i<allModelInfo.size(); i++ ){
            ModelInfoPersistable mip = allModelInfo.get(i);
            double currScore = mip.getScore();
            double t = (mip.getTimeStamp() - execStartTime) / 60000.0;    //60000 ms per minute

            allX[i] = t;
            allY[i] = currScore;

            if(i == 0){
                bestX.add(t);
                bestY.add(currScore);
                bestScore = currScore;
                bestModel = mip;
            } else if((!minimize && currScore > bestScore) || (minimize && currScore < bestScore)){
                bestX.add(t);
                bestY.add(bestScore);
                bestX.add(t);  //TODO non-real time rendering support...
                bestY.add(currScore);

                bestScore = currScore;
                bestModel = mip;
            }

            if((!minimize && currScore < worstScore) || (minimize && currScore > worstScore)){
                worstScore = currScore;
            }

            if(t > lastTime){
                lastTime = t;
            }
        }


        double[] scatterGraphMinMax = UIUtils.graphNiceRange(Math.max(bestScore, worstScore), Math.min(bestScore, worstScore), 5);
        double[] lineGraphMinMax = UIUtils.graphNiceRange(
                bestY.stream().mapToDouble(s -> s).max().orElse(0),bestY.stream().mapToDouble(s -> s).min().orElse(0), 5
        );

        if(bestX.size() > 0) {
            bestX.add(lastTime);
            bestY.add(bestY.get(bestY.size() - 1));
        }


        double[] bestXd = new double[bestX.size()];
        double[] bestYd = new double[bestXd.length];
        for( int i=0; i<bestX.size(); i++ ){
            bestXd[i] = bestX.get(i);
            bestYd[i] = bestY.get(i);
        }

        List<Component> components = new ArrayList<>(2);

        ChartLine cl = new ChartLine.Builder("Best Model Score vs. Time (Minutes)", STYLE_CHART_560_320)
                .addSeries("Best Score vs. Time", bestXd, bestYd)
                .setYMin(lineGraphMinMax[0])
                .setYMax(lineGraphMinMax[1])
                .build();
        components.add(cl);

        ChartScatter cs = new ChartScatter.Builder("All Candidate Scores vs. Time (Minutes)", STYLE_CHART_560_320)
                .addSeries("Candidates", allX, allY)
                .setYMin(scatterGraphMinMax[0])
                .setYMax(scatterGraphMinMax[1])
                .build();

        components.add(cs);

        return new Pair<>(components, bestModel);
    }
}
