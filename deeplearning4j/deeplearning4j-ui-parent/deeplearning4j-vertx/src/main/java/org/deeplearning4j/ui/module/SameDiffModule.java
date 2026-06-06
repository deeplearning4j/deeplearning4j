/*
 *  ******************************************************************************
 *  *
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

package org.deeplearning4j.ui.module;

import io.vertx.ext.web.FileUpload;
import io.vertx.ext.web.RoutingContext;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.core.storage.StatsStorageEvent;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.i18n.I18NResource;
import org.deeplearning4j.ui.module.samediff.SameDiffGraphSerializer;
import org.deeplearning4j.ui.module.train.TrainModuleUtils;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.File;
import java.util.*;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * SameDiff UI module at /samediff.
 * Provides graph upload, visualization, and info endpoints for SameDiff and DL4J models.
 */
@Slf4j
public class SameDiffModule implements UIModule {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private volatile Map<String, Object> currentGraph = null;
    private volatile Map<String, Object> currentInfo = null;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    public SameDiffModule() {
    }

    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.emptyList();
    }

    @Override
    public List<Route> getRoutes() {
        List<Route> routes = new ArrayList<>();
        routes.add(new Route("/samediff", HttpMethod.GET,
                (path, rc) -> rc.response().sendFile("templates/SameDiffUI.html")));
        routes.add(new Route("/samediff/upload", HttpMethod.POST,
                (path, rc) -> handleUpload(rc)));
        routes.add(new Route("/samediff/graph", HttpMethod.GET,
                (path, rc) -> getGraph(rc)));
        routes.add(new Route("/samediff/info", HttpMethod.GET,
                (path, rc) -> getInfo(rc)));
        return routes;
    }

    private void getGraph(RoutingContext rc) {
        lock.readLock().lock();
        try {
            if (currentGraph == null) {
                rc.response()
                        .putHeader("content-type", "application/json")
                        .end("{}");
            } else {
                rc.response()
                        .putHeader("content-type", "application/json")
                        .end(MAPPER.writeValueAsString(currentGraph));
            }
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        } finally {
            lock.readLock().unlock();
        }
    }

    private void getInfo(RoutingContext rc) {
        lock.readLock().lock();
        try {
            if (currentInfo == null) {
                Map<String, Object> noInfo = new LinkedHashMap<>();
                noInfo.put("loaded", false);
                rc.response()
                        .putHeader("content-type", "application/json")
                        .end(MAPPER.writeValueAsString(noInfo));
            } else {
                rc.response()
                        .putHeader("content-type", "application/json")
                        .end(MAPPER.writeValueAsString(currentInfo));
            }
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        } finally {
            lock.readLock().unlock();
        }
    }

    @SuppressWarnings("unchecked")
    private void handleUpload(RoutingContext rc) {
        Set<FileUpload> uploads = rc.fileUploads();
        if (uploads == null || uploads.isEmpty()) {
            rc.response().setStatusCode(400).end("{\"error\": \"No file uploaded\"}");
            return;
        }

        FileUpload upload = uploads.iterator().next();
        String filename = upload.fileName();
        String ext = FilenameUtils.getExtension(filename).toLowerCase();
        File file = new File(upload.uploadedFileName());

        try {
            long startTime = System.currentTimeMillis();
            Map<String, Object> graph;
            String modelType;

            switch (ext) {
                case "sdz":
                    SameDiff sdz = SameDiff.loadSdz(file);
                    graph = SameDiffGraphSerializer.serialize(sdz);
                    modelType = "SameDiff";
                    break;
                case "fb":
                case "flatbuffers":
                    SameDiff sdfb = SameDiff.fromFlatFile(file);
                    graph = SameDiffGraphSerializer.serialize(sdfb);
                    modelType = "SameDiff";
                    break;
                case "zip":
                    try {
                        ComputationGraph cg = ModelSerializer.restoreComputationGraph(file, false);
                        TrainModuleUtils.GraphInfo gi = TrainModuleUtils.buildGraphInfo(cg.getConfiguration());
                        graph = convertGraphInfoToNodes(gi, "ComputationGraph");
                        modelType = "ComputationGraph";
                    } catch (Exception e1) {
                        try {
                            MultiLayerNetwork mln = ModelSerializer.restoreMultiLayerNetwork(file, false);
                            TrainModuleUtils.GraphInfo gi = TrainModuleUtils.buildGraphInfo(mln.getLayerWiseConfigurations());
                            graph = convertGraphInfoToNodes(gi, "MultiLayerNetwork");
                            modelType = "MultiLayerNetwork";
                        } catch (Exception e2) {
                            throw new RuntimeException("Could not load as ComputationGraph or MultiLayerNetwork: " + e1.getMessage());
                        }
                    }
                    break;
                default:
                    try {
                        SameDiff sdGeneric = SameDiff.load(file, false);
                        graph = SameDiffGraphSerializer.serialize(sdGeneric);
                        modelType = "SameDiff";
                    } catch (Exception e) {
                        throw new RuntimeException("Unsupported model format: " + ext);
                    }
                    break;
            }

            long loadTime = System.currentTimeMillis() - startTime;

            Map<String, Object> info = new LinkedHashMap<>();
            info.put("loaded", true);
            info.put("filename", filename);
            info.put("modelType", modelType);
            info.put("opCount", graph.get("opCount"));
            info.put("varCount", graph.get("varCount"));
            info.put("paramCount", graph.get("paramCount"));
            info.put("loadTimeMs", loadTime);

            lock.writeLock().lock();
            try {
                currentGraph = graph;
                currentInfo = info;
            } finally {
                lock.writeLock().unlock();
            }

            // Return the full graph data plus info fields at top level
            Map<String, Object> response = new LinkedHashMap<>();
            response.putAll(graph);
            // Also include info fields at top level for direct access
            response.put("filename", filename);
            response.put("loadTimeMs", loadTime);

            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(response));

        } catch (Exception e) {
            log.error("Failed to load model file: {}", filename, e);
            rc.response().setStatusCode(500)
                    .end("{\"error\": \"Failed to load model: " + e.getMessage().replace("\"", "'") + "\"}");
        } finally {
            if (file.exists()) {
                file.delete();
            }
        }
    }

    private Map<String, Object> convertGraphInfoToNodes(TrainModuleUtils.GraphInfo gi, String modelType) {
        List<Map<String, Object>> nodes = new ArrayList<>();
        List<Map<String, Object>> edges = new ArrayList<>();
        int edgeId = 0;

        List<String> vertexNames = gi.getVertexNames();
        List<String> vertexTypes = gi.getVertexTypes();
        List<List<Integer>> vertexInputs = gi.getVertexInputs();
        List<Map<String, String>> vertexInfo = gi.getVertexInfo();

        for (int i = 0; i < vertexNames.size(); i++) {
            Map<String, Object> nodeData = new LinkedHashMap<>();
            nodeData.put("id", "layer-" + i);
            nodeData.put("label", vertexNames.get(i));
            nodeData.put("opType", vertexTypes.get(i));
            nodeData.put("nodeType", "op");

            Map<String, Object> props = new LinkedHashMap<>();
            if (vertexInfo.get(i) != null) {
                props.putAll(vertexInfo.get(i));
            }
            nodeData.put("properties", props);

            String cssClass = "op";
            String type = vertexTypes.get(i).toLowerCase();
            if (type.contains("input")) cssClass = "placeholder";
            else if (type.contains("output")) cssClass = "output";

            Map<String, Object> node = new LinkedHashMap<>();
            node.put("data", nodeData);
            node.put("classes", cssClass);
            nodes.add(node);

            List<Integer> inputs = vertexInputs.get(i);
            if (inputs != null) {
                for (int inputIdx : inputs) {
                    Map<String, Object> edgeData = new LinkedHashMap<>();
                    edgeData.put("id", "e" + (edgeId++));
                    edgeData.put("source", "layer-" + inputIdx);
                    edgeData.put("target", "layer-" + i);
                    Map<String, Object> edge = new LinkedHashMap<>();
                    edge.put("data", edgeData);
                    edges.add(edge);
                }
            }
        }

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("nodes", nodes);
        result.put("edges", edges);
        result.put("inputs", Collections.singletonList(vertexNames.get(0)));
        result.put("outputs", Collections.singletonList(vertexNames.get(vertexNames.size() - 1)));
        result.put("modelType", modelType);
        result.put("opCount", vertexNames.size());
        result.put("varCount", 0);
        result.put("paramCount", 0);
        return result;
    }

    @Override
    public void reportStorageEvents(Collection<StatsStorageEvent> events) {
    }

    @Override
    public void onAttach(StatsStorage statsStorage) {
    }

    @Override
    public void onDetach(StatsStorage statsStorage) {
    }

    @Override
    public List<I18NResource> getInternationalizationResources() {
        return Collections.emptyList();
    }
}
