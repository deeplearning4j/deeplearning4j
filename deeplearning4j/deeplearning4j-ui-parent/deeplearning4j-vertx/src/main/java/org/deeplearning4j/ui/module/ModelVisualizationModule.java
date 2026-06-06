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
import java.lang.reflect.Method;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Standalone model visualization module at /modelview.
 * Supports uploading and viewing SameDiff, ONNX, GGUF, ComputationGraph,
 * and MultiLayerNetwork models as interactive graph diagrams.
 */
@Slf4j
public class ModelVisualizationModule implements UIModule {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private volatile Map<String, Object> currentGraph = null;
    private volatile Map<String, Object> currentInfo = null;
    private final ConcurrentHashMap<String, Map<String, Object>> nodeDetails = new ConcurrentHashMap<>();
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.emptyList();
    }

    @Override
    public List<Route> getRoutes() {
        List<Route> routes = new ArrayList<>();
        routes.add(new Route("/modelview", HttpMethod.GET, (path, rc) -> rc.response().sendFile("templates/ModelViewer.html")));
        routes.add(new Route("/modelview/upload", HttpMethod.POST, (path, rc) -> handleUpload(rc)));
        routes.add(new Route("/modelview/current", HttpMethod.GET, (path, rc) -> getCurrent(rc)));
        routes.add(new Route("/modelview/node/:nodeId", HttpMethod.GET, (path, rc) -> getNodeDetails(rc)));
        return routes;
    }

    private void getCurrent(RoutingContext rc) {
        lock.readLock().lock();
        try {
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("graph", currentGraph);
            result.put("info", currentInfo);
            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(result));
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        } finally {
            lock.readLock().unlock();
        }
    }

    private void getNodeDetails(RoutingContext rc) {
        String nodeId = rc.pathParam("nodeId");
        Map<String, Object> details = nodeDetails.get(nodeId);
        try {
            if (details == null) {
                rc.response().setStatusCode(404).end("{\"error\": \"Node not found\"}");
            } else {
                rc.response()
                        .putHeader("content-type", "application/json")
                        .end(MAPPER.writeValueAsString(details));
            }
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
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
                    modelType = "SameDiff (.sdz)";
                    break;
                case "fb":
                case "flatbuffers":
                    SameDiff sdfb = SameDiff.fromFlatFile(file);
                    graph = SameDiffGraphSerializer.serialize(sdfb);
                    modelType = "SameDiff (.fb)";
                    break;
                case "onnx":
                    SameDiff sdOnnx = loadOnnxModel(file);
                    graph = SameDiffGraphSerializer.serialize(sdOnnx);
                    modelType = "ONNX (.onnx)";
                    break;
                case "gguf":
                    SameDiff sdGguf = loadGgufModel(file);
                    graph = SameDiffGraphSerializer.serialize(sdGguf);
                    modelType = "GGUF (.gguf)";
                    break;
                case "zip":
                    try {
                        ComputationGraph cg = ModelSerializer.restoreComputationGraph(file, false);
                        TrainModuleUtils.GraphInfo gi = TrainModuleUtils.buildGraphInfo(cg.getConfiguration());
                        graph = convertGraphInfoToNodes(gi, "ComputationGraph");
                        modelType = "ComputationGraph (.zip)";
                    } catch (Exception e1) {
                        try {
                            MultiLayerNetwork mln = ModelSerializer.restoreMultiLayerNetwork(file, false);
                            TrainModuleUtils.GraphInfo gi = TrainModuleUtils.buildGraphInfo(mln.getLayerWiseConfigurations());
                            graph = convertGraphInfoToNodes(gi, "MultiLayerNetwork");
                            modelType = "MultiLayerNetwork (.zip)";
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

            // Index node details for per-node lookup
            nodeDetails.clear();
            List<Map<String, Object>> nodes = (List<Map<String, Object>>) graph.get("nodes");
            if (nodes != null) {
                for (Map<String, Object> node : nodes) {
                    Map<String, Object> data = (Map<String, Object>) node.get("data");
                    if (data != null && data.get("id") != null) {
                        nodeDetails.put(String.valueOf(data.get("id")), data);
                    }
                }
            }

            lock.writeLock().lock();
            try {
                currentGraph = graph;
                currentInfo = info;
            } finally {
                lock.writeLock().unlock();
            }

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("graph", graph);
            response.put("info", info);
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

    private SameDiff loadOnnxModel(File file) {
        try {
            Class<?> importerClass = Class.forName("org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter");
            Object importer = importerClass.getDeclaredConstructor().newInstance();
            Method runImport = importerClass.getMethod("runImport", String.class, Map.class, boolean.class, boolean.class);
            return (SameDiff) runImport.invoke(importer, file.getAbsolutePath(), Collections.emptyMap(), false, false);
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("ONNX import not available — samediff-import-onnx module not on classpath");
        } catch (Exception e) {
            throw new RuntimeException("Failed to import ONNX model: " + e.getMessage(), e);
        }
    }

    private SameDiff loadGgufModel(File file) {
        try {
            Class<?> ggmlClass = Class.forName("org.nd4j.ggml.GGMLModelImport");
            Method importModel = ggmlClass.getMethod("importModel", String.class);
            return (SameDiff) importModel.invoke(null, file.getAbsolutePath());
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("GGUF import not available — nd4j-ggml module not on classpath");
        } catch (Exception e) {
            throw new RuntimeException("Failed to import GGUF model: " + e.getMessage(), e);
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
    public void reportStorageEvents(Collection<StatsStorageEvent> events) { }

    @Override
    public void onAttach(StatsStorage statsStorage) { }

    @Override
    public void onDetach(StatsStorage statsStorage) { }

    @Override
    public List<I18NResource> getInternationalizationResources() {
        return Collections.emptyList();
    }
}
