/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.ui.module.dsp;

import io.vertx.ext.web.RoutingContext;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.core.storage.Persistable;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.core.storage.StatsStorageEvent;
import org.deeplearning4j.core.storage.StatsStorageListener;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.i18n.I18NResource;
import org.deeplearning4j.ui.model.stats.dsp.DspDiagnosticsInitReport;
import org.deeplearning4j.ui.model.stats.dsp.DspDiagnosticsReport;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * UI module for DSP (Dynamic Shape Plan) diagnostics visualization.
 * Receives {@link DspDiagnosticsReport} updates from {@link org.deeplearning4j.ui.model.stats.DspStatsListener}
 * and serves them via REST endpoints for the DSP Dashboard frontend.
 */
@Slf4j
public class DspDiagModule implements UIModule {

    private static final String TYPE_ID = DspDiagnosticsReport.TYPE_ID;
    private static final ObjectMapper MAPPER = new ObjectMapper();

    // Cached state per session
    private final ConcurrentHashMap<String, DspDiagnosticsReport> latestReports = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, List<DspDiagnosticsReport>> reportHistory = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, DspDiagnosticsInitReport> initReports = new ConcurrentHashMap<>();
    private final Set<String> knownSessionIDs = Collections.synchronizedSet(new LinkedHashSet<>());

    // Storage references
    private final List<StatsStorage> attachedStorage = new CopyOnWriteArrayList<>();

    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.singletonList(TYPE_ID);
    }

    @Override
    public List<Route> getRoutes() {
        List<Route> routes = new ArrayList<>();
        routes.add(new Route("/dsp", HttpMethod.GET, (path, rc) -> rc.response().sendFile("templates/DspDashboard.html")));
        routes.add(new Route("/dsp/state", HttpMethod.GET, (path, rc) -> handleState(rc)));
        routes.add(new Route("/dsp/state/history", HttpMethod.GET, (path, rc) -> handleStateHistory(rc)));
        routes.add(new Route("/dsp/graph", HttpMethod.GET, (path, rc) -> handleGraph(rc)));
        routes.add(new Route("/dsp/segments", HttpMethod.GET, (path, rc) -> handleSegments(rc)));
        routes.add(new Route("/dsp/ops", HttpMethod.GET, (path, rc) -> handleOps(rc)));
        routes.add(new Route("/dsp/memory", HttpMethod.GET, (path, rc) -> handleMemory(rc)));
        routes.add(new Route("/dsp/info", HttpMethod.GET, (path, rc) -> handleInfo(rc)));
        routes.add(new Route("/dsp/sessions", HttpMethod.GET, (path, rc) -> handleSessions(rc)));
        return routes;
    }

    @Override
    public synchronized void reportStorageEvents(Collection<StatsStorageEvent> events) {
        for (StatsStorageEvent sse : events) {
            if (!TYPE_ID.equals(sse.getTypeID())) continue;

            String sid = sse.getSessionID();
            knownSessionIDs.add(sid);

            if (sse.getEventType() == StatsStorageListener.EventType.PostStaticInfo) {
                // Init report
                Persistable p = sse.getStatsStorage().getStaticInfo(sid, TYPE_ID, sse.getWorkerID());
                if (p instanceof DspDiagnosticsInitReport) {
                    initReports.put(sid, (DspDiagnosticsInitReport) p);
                }
            } else if (sse.getEventType() == StatsStorageListener.EventType.PostUpdate) {
                // Update report — read from storage
                List<Persistable> updates = sse.getStatsStorage().getAllUpdatesAfter(
                        sid, TYPE_ID, sse.getWorkerID(), 0);
                if (updates != null && !updates.isEmpty()) {
                    Persistable latest = updates.get(updates.size() - 1);
                    if (latest instanceof DspDiagnosticsReport) {
                        DspDiagnosticsReport dr = (DspDiagnosticsReport) latest;
                        latestReports.put(sid, dr);
                        reportHistory.computeIfAbsent(sid, k -> new CopyOnWriteArrayList<>()).add(dr);
                    }
                }
            }
        }
    }

    @Override
    public void onAttach(StatsStorage statsStorage) {
        attachedStorage.add(statsStorage);
        // Load any existing DSP data from this storage
        for (String sid : statsStorage.listSessionIDs()) {
            if (statsStorage.sessionExists(sid)) {
                List<Persistable> staticInfos = statsStorage.getAllStaticInfos(sid, TYPE_ID);
                if (staticInfos != null) {
                    for (Persistable p : staticInfos) {
                        if (p instanceof DspDiagnosticsInitReport) {
                            knownSessionIDs.add(sid);
                            initReports.put(sid, (DspDiagnosticsInitReport) p);
                        }
                    }
                }
            }
        }
    }

    @Override
    public void onDetach(StatsStorage statsStorage) {
        attachedStorage.remove(statsStorage);
    }

    @Override
    public List<I18NResource> getInternationalizationResources() {
        return Collections.emptyList();
    }

    // ─── HTTP Handlers ──────────────────────────────────────────────────

    private void handleState(RoutingContext rc) {
        String sid = getSessionID(rc);
        DspDiagnosticsReport report = sid != null ? latestReports.get(sid) : getAnyLatest();
        if (report == null) {
            rc.response().putHeader("content-type", "application/json")
                    .end("{\"error\":\"No DSP data available. Attach a DspStatsListener to a SameDiff model.\"}");
            return;
        }
        jsonResponse(rc, report);
    }

    private void handleStateHistory(RoutingContext rc) {
        String sid = getSessionID(rc);
        List<DspDiagnosticsReport> history = sid != null ? reportHistory.get(sid) : getAnyHistory();
        if (history == null || history.isEmpty()) {
            rc.response().putHeader("content-type", "application/json").end("[]");
            return;
        }
        jsonResponse(rc, history);
    }

    private void handleGraph(RoutingContext rc) {
        DspDiagnosticsReport report = getReport(rc);
        if (report == null || report.getDotGraph() == null) {
            rc.response().putHeader("content-type", "text/plain").end("");
            return;
        }
        rc.response().putHeader("content-type", "text/plain").end(report.getDotGraph());
    }

    private void handleSegments(RoutingContext rc) {
        DspDiagnosticsReport report = getReport(rc);
        if (report == null || report.getSegments() == null) {
            rc.response().putHeader("content-type", "application/json").end("[]");
            return;
        }
        jsonResponse(rc, report.getSegments());
    }

    private void handleOps(RoutingContext rc) {
        DspDiagnosticsReport report = getReport(rc);
        if (report == null || report.getOpHistogram() == null) {
            rc.response().putHeader("content-type", "application/json").end("{}");
            return;
        }
        jsonResponse(rc, report.getOpHistogram());
    }

    private void handleMemory(RoutingContext rc) {
        DspDiagnosticsReport report = getReport(rc);
        if (report == null || report.getMemoryTimeline() == null) {
            rc.response().putHeader("content-type", "application/json").end("[]");
            return;
        }
        jsonResponse(rc, report.getMemoryTimeline());
    }

    private void handleInfo(RoutingContext rc) {
        String sid = getSessionID(rc);
        DspDiagnosticsInitReport init = sid != null ? initReports.get(sid) : getAnyInit();
        if (init == null) {
            rc.response().putHeader("content-type", "application/json")
                    .end("{\"error\":\"No DSP init data available\"}");
            return;
        }
        jsonResponse(rc, init);
    }

    private void handleSessions(RoutingContext rc) {
        jsonResponse(rc, new ArrayList<>(knownSessionIDs));
    }

    // ─── Helpers ────────────────────────────────────────────────────────

    private String getSessionID(RoutingContext rc) {
        String sid = rc.request().getParam("session");
        return sid != null && !sid.isEmpty() ? sid : null;
    }

    private DspDiagnosticsReport getReport(RoutingContext rc) {
        String sid = getSessionID(rc);
        return sid != null ? latestReports.get(sid) : getAnyLatest();
    }

    private DspDiagnosticsReport getAnyLatest() {
        if (latestReports.isEmpty()) return null;
        return latestReports.values().iterator().next();
    }

    private List<DspDiagnosticsReport> getAnyHistory() {
        if (reportHistory.isEmpty()) return null;
        return reportHistory.values().iterator().next();
    }

    private DspDiagnosticsInitReport getAnyInit() {
        if (initReports.isEmpty()) return null;
        return initReports.values().iterator().next();
    }

    private void jsonResponse(RoutingContext rc, Object data) {
        try {
            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(data));
        } catch (Exception e) {
            rc.response().setStatusCode(500)
                    .end("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }
}
