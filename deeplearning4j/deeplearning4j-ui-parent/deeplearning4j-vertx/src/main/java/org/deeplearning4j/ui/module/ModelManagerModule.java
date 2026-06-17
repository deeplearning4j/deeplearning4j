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

import io.vertx.ext.web.RoutingContext;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.config.DL4JSystemProperties;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.core.storage.StatsStorageEvent;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.i18n.I18NResource;
import org.deeplearning4j.ui.module.modelmanager.*;
import org.deeplearning4j.ui.module.samediff.SameDiffGraphSerializer;
import org.deeplearning4j.ui.module.train.TrainModuleUtils;
import org.deeplearning4j.util.ModelSerializer;
import org.eclipse.deeplearning4j.omnihub.OmniHubUtils;
import org.eclipse.deeplearning4j.omnihub.repository.ModelRepositoryRegistry;
import org.eclipse.deeplearning4j.pipeline.AutoModel;
import org.eclipse.deeplearning4j.pipeline.PipelineLoader;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanDiskCache;
import org.nd4j.autodiff.samediff.execution.TritonCacheManager;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.*;

/**
 * UI module for browsing a pre-defined model catalog, downloading models
 * from HuggingFace, and managing locally installed models.
 *
 * Handles tokenizer configuration, model metadata extraction (architecture,
 * quantization, context length from GGUF headers), and chat template discovery.
 *
 * Routes:
 *   GET  /models                         - Serve ModelManager.html
 *   GET  /models/catalog                 - Catalog JSON with installed status + LLM metadata
 *   GET  /models/installed               - Registry entries
 *   POST /models/download                - Start async download, return downloadId
 *   GET  /models/download/:id/progress   - SSE progress stream
 *   POST /models/remove/:id              - Remove model
 *   GET  /models/load/:id                - Load model graph
 *   POST /models/omnihub/download        - Download via OmniHub
 *   GET  /models/:id/info                - Full model info (metadata + tokenizer)
 *   GET  /models/:id/tokenizer           - Tokenizer configuration
 *   GET  /models/:id/chat-template       - Chat template string
 *   GET  /models/cache/status            - Overview of all caches (DSP, Triton, SDZ)
 *   GET  /models/cache/dsp               - List DSP plan cache entries with metadata
 *   POST /models/cache/dsp/clear         - Clear all DSP plan cache entries
 *   POST /models/cache/dsp/:hash         - Invalidate a single DSP plan entry
 *   GET  /models/cache/triton            - Triton kernel cache info and listing
 *   POST /models/cache/triton/clear      - Clear all Triton kernel cache
 *   POST /models/cache/triton/export     - Export Triton cache to .tkcache bundle
 *   POST /models/cache/triton/import     - Import .tkcache bundle
 *   GET  /models/cache/sdz               - List cached SDZ model files
 *   POST /models/cache/sdz/clear         - Clear all cached SDZ files
 *   POST /models/cache/sdz/:filename    - Evict a single cached SDZ file
 */
@Slf4j
public class ModelManagerModule implements UIModule {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final ModelRegistry registry;
    private final ModelCatalogService catalog;
    private final HuggingFaceDownloader downloader;
    private final ModelRepositoryRegistry omniHubRegistry;
    private final Path sdzCacheDir;
    private final ExecutorService downloadExecutor = Executors.newFixedThreadPool(2);
    private final ConcurrentHashMap<String, DownloadState> activeDownloads = new ConcurrentHashMap<>();

    private static class DownloadState {
        volatile String status = "downloading";
        volatile int percent = 0;
        volatile String error = null;
        volatile String fileName = "";
        volatile long bytesDownloaded = 0;
        volatile long totalBytes = 0;
        final List<RoutingContext> sseClients = new CopyOnWriteArrayList<>();
    }

    public ModelManagerModule() {
        this.registry = new ModelRegistry();
        this.catalog = new ModelCatalogService(registry);
        String token = System.getProperty(DL4JSystemProperties.HF_TOKEN, System.getenv("HF_TOKEN"));
        this.downloader = new HuggingFaceDownloader(token);
        this.omniHubRegistry = ModelRepositoryRegistry.getDefault();
        this.sdzCacheDir = Paths.get(System.getProperty("user.home"), ".cache", "samediff", "models");
        try {
            Files.createDirectories(sdzCacheDir);
        } catch (Exception e) {
            log.warn("Could not create SDZ cache directory: {}", sdzCacheDir, e);
        }
    }

    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.emptyList();
    }

    @Override
    public List<Route> getRoutes() {
        List<Route> routes = new ArrayList<>();
        routes.add(new Route("/models", HttpMethod.GET, (path, rc) -> rc.response().sendFile("templates/ModelManager.html")));
        routes.add(new Route("/models/catalog", HttpMethod.GET, (path, rc) -> handleCatalog(rc)));
        routes.add(new Route("/models/installed", HttpMethod.GET, (path, rc) -> handleInstalled(rc)));
        routes.add(new Route("/models/download", HttpMethod.POST, (path, rc) -> handleDownload(rc)));
        routes.add(new Route("/models/download/:id/progress", HttpMethod.GET, (path, rc) -> handleProgress(rc)));
        routes.add(new Route("/models/remove/:id", HttpMethod.POST, (path, rc) -> handleRemove(rc)));
        routes.add(new Route("/models/load/:id", HttpMethod.GET, (path, rc) -> handleLoadModel(rc)));
        routes.add(new Route("/models/omnihub/download", HttpMethod.POST, (path, rc) -> handleOmniHubDownload(rc)));
        // Model info, tokenizer, and chat template endpoints
        routes.add(new Route("/models/:id/info", HttpMethod.GET, (path, rc) -> handleGetInfo(rc)));
        routes.add(new Route("/models/:id/tokenizer", HttpMethod.GET, (path, rc) -> handleGetTokenizer(rc)));
        routes.add(new Route("/models/:id/chat-template", HttpMethod.GET, (path, rc) -> handleGetChatTemplate(rc)));
        // Cache management endpoints
        routes.add(new Route("/models/cache/status", HttpMethod.GET, (path, rc) -> handleCacheStatus(rc)));
        routes.add(new Route("/models/cache/dsp", HttpMethod.GET, (path, rc) -> handleDspCacheList(rc)));
        routes.add(new Route("/models/cache/dsp/clear", HttpMethod.POST, (path, rc) -> handleDspCacheClear(rc)));
        routes.add(new Route("/models/cache/dsp/:hash", HttpMethod.POST, (path, rc) -> handleDspCacheInvalidate(rc)));
        routes.add(new Route("/models/cache/triton", HttpMethod.GET, (path, rc) -> handleTritonCacheInfo(rc)));
        routes.add(new Route("/models/cache/triton/clear", HttpMethod.POST, (path, rc) -> handleTritonCacheClear(rc)));
        routes.add(new Route("/models/cache/triton/export", HttpMethod.POST, (path, rc) -> handleTritonCacheExport(rc)));
        routes.add(new Route("/models/cache/triton/import", HttpMethod.POST, (path, rc) -> handleTritonCacheImport(rc)));
        routes.add(new Route("/models/cache/sdz", HttpMethod.GET, (path, rc) -> handleSdzCacheList(rc)));
        routes.add(new Route("/models/cache/sdz/clear", HttpMethod.POST, (path, rc) -> handleSdzCacheClear(rc)));
        routes.add(new Route("/models/cache/sdz/:filename", HttpMethod.POST, (path, rc) -> handleSdzCacheEvict(rc)));
        return routes;
    }

    // ── Catalog & Registry ──────────────────────────────────────────────

    private void handleCatalog(RoutingContext rc) {
        try {
            List<ModelCatalogService.CatalogEntry> entries = catalog.getCatalog();
            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(entries));
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    private void handleInstalled(RoutingContext rc) {
        try {
            List<ModelRegistryEntry> entries = registry.listEntries();
            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(entries));
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    // ── Model Info, Tokenizer, Chat Template ────────────────────────────

    private void handleGetInfo(RoutingContext rc) {
        String id = rc.pathParam("id");
        try {
            ModelRegistryEntry entry = registry.getEntry(id);
            if (entry == null) {
                rc.response().setStatusCode(404).end("{\"error\": \"Model not found: " + id + "\"}");
                return;
            }

            Map<String, Object> info = new LinkedHashMap<>();
            info.put("id", entry.getId());
            info.put("status", entry.getStatus());
            info.put("format", entry.getFormat());
            info.put("source", entry.getSource());
            info.put("repo", entry.getRepo());
            info.put("localPath", entry.getLocalPath());
            info.put("fileSizeBytes", entry.getFileSizeBytes());
            info.put("checksum", entry.getChecksum());
            info.put("downloadedAt", entry.getDownloadedAt());
            info.put("modelType", entry.getModelType());
            info.put("metadata", entry.getMetadata());
            info.put("tokenizer", entry.getTokenizer());
            info.put("files", entry.getFiles());
            info.put("sdzCached", entry.isSdzCached());
            info.put("sdzPath", entry.getSdzPath());

            // If tokenizer_config.json exists on disk, include its raw contents
            if (entry.getLocalPath() != null) {
                Path modelDir = Paths.get(entry.getLocalPath()).getParent();
                if (modelDir != null) {
                    Path tokConfig = modelDir.resolve("tokenizer_config.json");
                    if (Files.exists(tokConfig)) {
                        String raw = Files.readString(tokConfig);
                        info.put("tokenizerConfigRaw", MAPPER.readValue(raw, Map.class));
                    }
                }
            }

            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(info));
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    private void handleGetTokenizer(RoutingContext rc) {
        String id = rc.pathParam("id");
        try {
            ModelRegistryEntry entry = registry.getEntry(id);
            if (entry == null) {
                rc.response().setStatusCode(404).end("{\"error\": \"Model not found: " + id + "\"}");
                return;
            }

            Map<String, Object> result = new LinkedHashMap<>();

            // Include stored tokenizer config from registry
            if (entry.getTokenizer() != null) {
                result.put("config", entry.getTokenizer());
            }

            // Try to read tokenizer_config.json from disk
            if (entry.getLocalPath() != null) {
                Path modelDir = Paths.get(entry.getLocalPath()).getParent();
                if (modelDir != null) {
                    Path tokConfig = modelDir.resolve("tokenizer_config.json");
                    if (Files.exists(tokConfig)) {
                        result.put("tokenizerConfig", MAPPER.readValue(Files.readString(tokConfig), Map.class));
                    }
                    Path specialTokens = modelDir.resolve("special_tokens_map.json");
                    if (Files.exists(specialTokens)) {
                        result.put("specialTokensMap", MAPPER.readValue(Files.readString(specialTokens), Map.class));
                    }
                }
            }

            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(result));
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    private void handleGetChatTemplate(RoutingContext rc) {
        String id = rc.pathParam("id");
        try {
            ModelRegistryEntry entry = registry.getEntry(id);
            if (entry == null) {
                rc.response().setStatusCode(404).end("{\"error\": \"Model not found: " + id + "\"}");
                return;
            }

            String chatTemplate = null;

            // 1. Check registry tokenizer config
            if (entry.getTokenizer() != null && entry.getTokenizer().getChatTemplate() != null) {
                chatTemplate = entry.getTokenizer().getChatTemplate();
            }

            // 2. Fall back to tokenizer_config.json on disk
            if (chatTemplate == null && entry.getLocalPath() != null) {
                Path modelDir = Paths.get(entry.getLocalPath()).getParent();
                if (modelDir != null) {
                    Path tokConfig = modelDir.resolve("tokenizer_config.json");
                    if (Files.exists(tokConfig)) {
                        Map<String, Object> config = MAPPER.readValue(Files.readString(tokConfig), Map.class);
                        Object ct = config.get("chat_template");
                        if (ct instanceof String) {
                            chatTemplate = (String) ct;
                        }
                    }
                }
            }

            // 3. Fall back to GGUF embedded chat template
            if (chatTemplate == null && entry.getLocalPath() != null && "gguf".equals(entry.getFormat())) {
                chatTemplate = extractGgufChatTemplate(new File(entry.getLocalPath()));
            }

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("id", id);
            result.put("chatTemplate", chatTemplate);
            result.put("chatTemplateFormat", entry.getTokenizer() != null ?
                    entry.getTokenizer().getChatTemplateFormat() : null);

            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(result));
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    // ── Download ────────────────────────────────────────────────────────

    private void handleDownload(RoutingContext rc) {
        try {
            Map<String, Object> body = MAPPER.readValue(rc.getBodyAsString(), Map.class);
            String modelId = (String) body.get("id");
            String repo = (String) body.get("repo");
            String filePath = (String) body.get("filePath");
            String format = (String) body.getOrDefault("format", "onnx");
            String description = (String) body.getOrDefault("description", "");

            if (modelId == null || repo == null || filePath == null) {
                rc.response().setStatusCode(400).end("{\"error\": \"Missing id, repo, or filePath\"}");
                return;
            }

            if (activeDownloads.containsKey(modelId)) {
                rc.response().setStatusCode(409).end("{\"error\": \"Already downloading\"}");
                return;
            }

            String downloadId = modelId;
            DownloadState state = new DownloadState();
            activeDownloads.put(downloadId, state);

            // Determine model type from format
            ModelRegistryEntry.ModelType modelType = inferModelType(format, body);

            String source = (String) body.getOrDefault("source", "huggingface");

            ModelRegistryEntry entry = ModelRegistryEntry.builder()
                    .id(modelId)
                    .source(source)
                    .repo(repo)
                    .format(format)
                    .status(ModelRegistryEntry.ModelStatus.DOWNLOADING)
                    .description(description)
                    .modelType(modelType)
                    .build();
            registry.addEntry(entry);

            downloadExecutor.submit(() -> {
                try {
                    File resolvedFile;
                    if ("omnihub".equals(source)) {
                        state.percent = 10;
                        state.fileName = modelId;
                        notifySseClients(downloadId, state);

                        String framework = "samediff";
                        if ("zip".equals(format)) framework = "dl4j";

                        if (repo.contains("/")) {
                            resolvedFile = omniHubRegistry.resolveHuggingFace(repo, filePath, false);
                        } else {
                            resolvedFile = omniHubRegistry.resolve(filePath != null ? filePath : modelId, framework, false);
                        }
                    } else {
                        Path destDir = registry.getRegistryDir().resolve(modelId);
                        HuggingFaceDownloader.DownloadResult result = downloader.download(
                                repo, filePath, destDir,
                                progress -> {
                                    state.percent = progress.getPercent();
                                    state.fileName = progress.getFileName();
                                    state.bytesDownloaded = progress.getBytesDownloaded();
                                    state.totalBytes = progress.getTotalBytes();
                                    notifySseClients(downloadId, state);
                                }
                        );
                        resolvedFile = result.getFilePath().toFile();
                        entry.setFileSizeBytes(result.getFileSize());
                        entry.setChecksum(result.getChecksum());

                        // Auto-fetch tokenizer files (best-effort)
                        Map<String, Path> tokenizerFiles = downloader.fetchTokenizerFiles(repo, "main", destDir);
                        if (!tokenizerFiles.isEmpty()) {
                            Map<String, String> files = new LinkedHashMap<>();
                            files.put("model", resolvedFile.getName());
                            for (Map.Entry<String, Path> tf : tokenizerFiles.entrySet()) {
                                files.put(tf.getKey(), tf.getValue().getFileName().toString());
                            }
                            entry.setFiles(files);
                        }
                    }

                    // Update registry with resolved path
                    entry.setLocalPath(resolvedFile.getAbsolutePath());
                    if (entry.getFileSizeBytes() == 0 && resolvedFile.isFile()) {
                        entry.setFileSizeBytes(resolvedFile.length());
                    }
                    entry.setFilename(resolvedFile.getName());
                    entry.setDownloadedAt(System.currentTimeMillis());
                    entry.setStatus(ModelRegistryEntry.ModelStatus.READY);

                    // Extract metadata for GGUF files
                    if ("gguf".equals(format)) {
                        extractAndPopulateGgufMetadata(entry, resolvedFile);
                    }

                    registry.addEntry(entry);

                    state.status = "complete";
                    state.percent = 100;
                    notifySseClients(downloadId, state);

                    // Import and cache as SDZ in background (non-blocking)
                    downloadExecutor.submit(() -> importAndCacheSDZ(entry));

                } catch (Exception e) {
                    log.error("Download failed for {}", modelId, e);
                    state.status = "error";
                    state.error = e.getMessage();
                    registry.updateStatus(modelId, ModelRegistryEntry.ModelStatus.ERROR);
                    notifySseClients(downloadId, state);
                } finally {
                    for (RoutingContext client : state.sseClients) {
                        try { client.response().end(); } catch (Exception ignored) {}
                    }
                    downloadExecutor.submit(() -> {
                        try { Thread.sleep(30000); } catch (InterruptedException ignored) {}
                        activeDownloads.remove(downloadId);
                    });
                }
            });

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("downloadId", downloadId);
            response.put("status", "started");
            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(response));

        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    private void handleProgress(RoutingContext rc) {
        String id = rc.pathParam("id");
        DownloadState state = activeDownloads.get(id);

        rc.response()
                .putHeader("Content-Type", "text/event-stream")
                .putHeader("Cache-Control", "no-cache")
                .putHeader("Connection", "keep-alive")
                .setChunked(true);

        if (state == null) {
            ModelRegistryEntry entry = registry.getEntry(id);
            if (entry != null && entry.getStatus() == ModelRegistryEntry.ModelStatus.READY) {
                rc.response().write("data: {\"status\":\"complete\",\"percent\":100}\n\n");
            } else if (entry != null && entry.getStatus() == ModelRegistryEntry.ModelStatus.ERROR) {
                rc.response().write("data: {\"status\":\"error\",\"error\":\"Download failed\"}\n\n");
            } else {
                rc.response().write("data: {\"status\":\"not_found\"}\n\n");
            }
            rc.response().end();
            return;
        }

        sendSseEvent(rc, state);
        state.sseClients.add(rc);
        rc.response().closeHandler(v -> state.sseClients.remove(rc));
    }

    private void handleRemove(RoutingContext rc) {
        String id = rc.pathParam("id");
        try {
            ModelRegistryEntry entry = registry.removeEntry(id);
            if (entry == null) {
                rc.response().setStatusCode(404).end("{\"error\": \"Model not found\"}");
                return;
            }

            if (entry.getLocalPath() != null) {
                try {
                    Path modelFile = Paths.get(entry.getLocalPath());
                    Files.deleteIfExists(modelFile);
                    Path parent = modelFile.getParent();
                    if (parent != null) {
                        String[] remaining = parent.toFile().list();
                        if (remaining != null && remaining.length == 0) {
                            Files.deleteIfExists(parent);
                        }
                    }
                } catch (Exception e) {
                    log.warn("Could not delete model file for {}: {}", id, e.getMessage());
                }
            }

            // Clean up cached SDZ file
            if (entry.isSdzCached() && entry.getSdzPath() != null) {
                try {
                    Files.deleteIfExists(Paths.get(entry.getSdzPath()));
                } catch (Exception e) {
                    log.warn("Could not delete cached SDZ for {}: {}", id, e.getMessage());
                }
            }

            rc.response()
                    .putHeader("content-type", "application/json")
                    .end("{\"removed\": \"" + id + "\"}");
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    private void handleLoadModel(RoutingContext rc) {
        String id = rc.pathParam("id");
        try {
            ModelRegistryEntry entry = registry.getEntry(id);
            if (entry == null) {
                rc.response().setStatusCode(404).end("{\"error\": \"Model not found in registry\"}");
                return;
            }
            if (entry.getStatus() != ModelRegistryEntry.ModelStatus.READY) {
                rc.response().setStatusCode(400)
                        .end("{\"error\": \"Model is not ready (status: " + entry.getStatus() + ")\"}");
                return;
            }
            if (entry.getLocalPath() == null || !Files.exists(Paths.get(entry.getLocalPath()))) {
                rc.response().setStatusCode(404)
                        .end("{\"error\": \"Model file not found at: " + entry.getLocalPath() + "\"}");
                return;
            }

            File modelFile = new File(entry.getLocalPath());
            Map<String, Object> graph;
            boolean loadedFromSdz = false;

            // Check for cached SDZ first — fast path
            if (entry.isSdzCached() && entry.getSdzPath() != null) {
                File sdzFile = new File(entry.getSdzPath());
                if (sdzFile.exists()) {
                    SameDiff sd = null;
                    try {
                        sd = SameDiff.loadSdz(sdzFile);
                        graph = SameDiffGraphSerializer.serialize(sd);

                        Map<String, Object> result = new LinkedHashMap<>();
                        result.put("id", id);
                        result.put("graph", graph);
                        result.put("format", "sdz");
                        result.put("filename", entry.getFilename());
                        result.put("loadedFromCache", true);
                        result.put("sdzPath", entry.getSdzPath());
                        rc.response()
                                .putHeader("content-type", "application/json")
                                .end(MAPPER.writeValueAsString(result));
                        return;
                    } catch (Exception e) {
                        log.warn("Cached SDZ load failed for {}, falling back to raw format", id, e);
                    } finally {
                        if (sd != null) { try { sd.close(); } catch (Exception ignored) {} }
                    }
                }
            }

            if (modelFile.isDirectory()) {
                // Use AutoModel pipeline — includes SDZ caching
                PipelineLoader.LoadConfig config = PipelineLoader.LoadConfig.builder()
                        .cacheConvertedModel(true)
                        .cacheDirectory(sdzCacheDir.toFile())
                        .build();
                SameDiff sd = null;
                try {
                    sd = AutoModel.fromPretrained(modelFile, config);
                    graph = SameDiffGraphSerializer.serialize(sd);
                    // Record the SDZ cache path if not already set
                    if (!entry.isSdzCached()) {
                        String dirName = modelFile.getName();
                        String hash = Integer.toHexString(modelFile.getAbsolutePath().hashCode());
                        Path cachedSdz = sdzCacheDir.resolve(dirName + "_" + hash + ".sdz");
                        if (Files.exists(cachedSdz)) {
                            entry.setSdzPath(cachedSdz.toAbsolutePath().toString());
                            entry.setSdzCached(true);
                            registry.addEntry(entry);
                        }
                    }
                } finally {
                    if (sd != null) { try { sd.close(); } catch (Exception ignored) {} }
                }
            } else {
                String ext = FilenameUtils.getExtension(modelFile.getName()).toLowerCase();
                SameDiff sd = null;
                try {
                    switch (ext) {
                        case "sdz":
                            sd = SameDiff.loadSdz(modelFile);
                            graph = SameDiffGraphSerializer.serialize(sd);
                            break;
                        case "fb":
                        case "flatbuffers":
                            sd = SameDiff.fromFlatFile(modelFile);
                            graph = SameDiffGraphSerializer.serialize(sd);
                            break;
                        case "onnx":
                            sd = loadOnnxModel(modelFile);
                            graph = SameDiffGraphSerializer.serialize(sd);
                            break;
                        case "gguf":
                            sd = loadGgufModel(modelFile);
                            graph = SameDiffGraphSerializer.serialize(sd);
                            break;
                        case "zip":
                            try {
                                ComputationGraph cg = ModelSerializer.restoreComputationGraph(modelFile, false);
                                graph = convertGraphInfo(TrainModuleUtils.buildGraphInfo(cg.getConfiguration()), "ComputationGraph");
                            } catch (Exception e1) {
                                MultiLayerNetwork mln = ModelSerializer.restoreMultiLayerNetwork(modelFile, false);
                                graph = convertGraphInfo(TrainModuleUtils.buildGraphInfo(mln.getLayerWiseConfigurations()), "MultiLayerNetwork");
                            }
                            break;
                        default:
                            try {
                                sd = SameDiff.load(modelFile, false);
                                graph = SameDiffGraphSerializer.serialize(sd);
                            } catch (Exception e) {
                                rc.response().setStatusCode(400)
                                        .end("{\"error\": \"Unsupported format: " + ext + "\"}");
                                return;
                            }
                    }
                } finally {
                    if (sd != null) { try { sd.close(); } catch (Exception ignored) {} }
                }

                // If we loaded from raw format and don't have a cached SDZ yet, cache it now
                if (!entry.isSdzCached()) {
                    downloadExecutor.submit(() -> importAndCacheSDZ(entry));
                }
            }

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("id", id);
            result.put("graph", graph);
            result.put("format", entry.getFormat());
            result.put("filename", entry.getFilename());
            result.put("loadedFromCache", false);
            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(result));

        } catch (Exception e) {
            log.error("Failed to load model {}", id, e);
            rc.response().setStatusCode(500)
                    .end("{\"error\": \"Failed to load model: " + e.getMessage() + "\"}");
        }
    }

    private void handleOmniHubDownload(RoutingContext rc) {
        try {
            Map<String, Object> body = MAPPER.readValue(rc.getBodyAsString(), Map.class);
            String huggingFaceId = (String) body.get("huggingFaceId");
            String filePattern = (String) body.get("filePattern");
            String modelName = (String) body.get("name");

            if (huggingFaceId == null) {
                rc.response().setStatusCode(400).end("{\"error\": \"Missing huggingFaceId\"}");
                return;
            }

            String modelId = modelName != null ? modelName : huggingFaceId.replace("/", "-");

            if (activeDownloads.containsKey(modelId)) {
                rc.response().setStatusCode(409).end("{\"error\": \"Already downloading\"}");
                return;
            }

            DownloadState state = new DownloadState();
            activeDownloads.put(modelId, state);

            String format = filePattern != null && filePattern.contains("gguf") ? "gguf" : "safetensors";
            ModelRegistryEntry entry = ModelRegistryEntry.builder()
                    .id(modelId)
                    .source("omnihub")
                    .repo(huggingFaceId)
                    .format(format)
                    .status(ModelRegistryEntry.ModelStatus.DOWNLOADING)
                    .description("OmniHub: " + huggingFaceId)
                    .modelType(format.equals("gguf") ? ModelRegistryEntry.ModelType.LLM_GGUF : ModelRegistryEntry.ModelType.LLM_SAMEDIFF)
                    .build();
            registry.addEntry(entry);

            downloadExecutor.submit(() -> {
                try {
                    state.percent = 10;
                    state.fileName = huggingFaceId;
                    notifySseClients(modelId, state);

                    File resolvedDir = omniHubRegistry.resolveHuggingFace(huggingFaceId, filePattern, false);

                    state.percent = 90;
                    notifySseClients(modelId, state);

                    entry.setLocalPath(resolvedDir.getAbsolutePath());
                    if (resolvedDir.isFile()) {
                        entry.setFileSizeBytes(resolvedDir.length());
                    }
                    entry.setFilename(resolvedDir.getName());
                    entry.setDownloadedAt(System.currentTimeMillis());
                    entry.setStatus(ModelRegistryEntry.ModelStatus.READY);

                    // Extract GGUF metadata if applicable
                    if ("gguf".equals(format) && resolvedDir.isFile()) {
                        extractAndPopulateGgufMetadata(entry, resolvedDir);
                    }

                    registry.addEntry(entry);

                    state.status = "complete";
                    state.percent = 100;
                    notifySseClients(modelId, state);

                    // Import and cache as SDZ in background (non-blocking)
                    downloadExecutor.submit(() -> importAndCacheSDZ(entry));

                } catch (Exception e) {
                    log.error("OmniHub download failed for {}", huggingFaceId, e);
                    state.status = "error";
                    state.error = e.getMessage();
                    registry.updateStatus(modelId, ModelRegistryEntry.ModelStatus.ERROR);
                    notifySseClients(modelId, state);
                } finally {
                    for (RoutingContext client : state.sseClients) {
                        try { client.response().end(); } catch (Exception ignored) {}
                    }
                    downloadExecutor.submit(() -> {
                        try { Thread.sleep(30000); } catch (InterruptedException ignored) {}
                        activeDownloads.remove(modelId);
                    });
                }
            });

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("downloadId", modelId);
            response.put("status", "started");
            response.put("source", "omnihub");
            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(response));
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    // ── Import & SDZ Caching ─────────────────────────────────────────────

    /**
     * Import a downloaded model and cache the result as an SDZ file.
     * Follows the AutoModel.fromPretrained pattern:
     *   - For directories: uses AutoModel (detects format, imports, caches SDZ)
     *   - For GGUF files: uses GGMLModelImport.convertToSDZ via reflection
     *   - For ONNX files: imports to SameDiff, saves as SDZ
     *   - For SDZ files: already in target format, just records the path
     *
     * Updates the registry entry with sdzPath and sdzCached=true on success.
     */
    private void importAndCacheSDZ(ModelRegistryEntry entry) {
        String localPath = entry.getLocalPath();
        if (localPath == null) return;

        File modelFile = new File(localPath);
        if (!modelFile.exists()) return;

        try {
            Path sdzFile;

            if (modelFile.isDirectory()) {
                // Directory-based model: use AutoModel pipeline which handles SDZ caching internally
                PipelineLoader.LoadConfig config = PipelineLoader.LoadConfig.builder()
                        .cacheConvertedModel(true)
                        .cacheDirectory(sdzCacheDir.toFile())
                        .build();
                SameDiff sd = AutoModel.fromPretrained(modelFile, config);
                // AutoModel caches to ~/.cache/samediff/models/{name}_{hash}.sdz
                String dirName = modelFile.getName();
                String hash = Integer.toHexString(modelFile.getAbsolutePath().hashCode());
                sdzFile = sdzCacheDir.resolve(dirName + "_" + hash + ".sdz");
                sd.close();
            } else {
                String ext = FilenameUtils.getExtension(modelFile.getName()).toLowerCase();
                sdzFile = sdzCacheDir.resolve(entry.getId() + ".sdz");

                if ("sdz".equals(ext)) {
                    // Already SDZ — just point to it
                    entry.setSdzPath(modelFile.getAbsolutePath());
                    entry.setSdzCached(true);
                    registry.addEntry(entry);
                    return;
                } else if ("gguf".equals(ext)) {
                    // GGUF → SDZ via GGMLModelImport.convertToSDZ
                    try {
                        Class<?> cls = Class.forName("org.nd4j.ggml.GGMLModelImport");
                        Method m = cls.getMethod("convertToSDZ", File.class, File.class);
                        m.invoke(null, modelFile, sdzFile.toFile());
                    } catch (ClassNotFoundException e) {
                        log.info("nd4j-ggml not on classpath, skipping SDZ conversion for {}", entry.getId());
                        return;
                    }
                } else if ("onnx".equals(ext)) {
                    // ONNX → import → save SDZ
                    SameDiff sd = loadOnnxModel(modelFile);
                    SDZSerializer.save(sd, sdzFile.toFile(), false, null);
                    sd.close();
                } else if ("fb".equals(ext) || "flatbuffers".equals(ext)) {
                    SameDiff sd = SameDiff.fromFlatFile(modelFile);
                    SDZSerializer.save(sd, sdzFile.toFile(), false, null);
                    sd.close();
                } else {
                    log.debug("No SDZ conversion available for format: {}", ext);
                    return;
                }
            }

            if (Files.exists(sdzFile)) {
                entry.setSdzPath(sdzFile.toAbsolutePath().toString());
                entry.setSdzCached(true);
                registry.addEntry(entry);
                log.info("Cached SDZ for model {}: {}", entry.getId(), sdzFile);
            }
        } catch (Exception e) {
            log.warn("SDZ import/cache failed for {}: {}", entry.getId(), e.getMessage());
        }
    }

    // ── GGUF Metadata Extraction ────────────────────────────────────────

    /**
     * Extract metadata from a GGUF file and populate the registry entry's
     * metadata and tokenizer config. Uses GGMLModelImport.inspectModel()
     * which reads only the header (no tensor data loaded).
     */
    private void extractAndPopulateGgufMetadata(ModelRegistryEntry entry, File ggufFile) {
        try {
            Class<?> importClass = Class.forName("org.nd4j.ggml.GGMLModelImport");
            Method inspectMethod = importClass.getMethod("inspectModel", File.class);
            Object metadata = inspectMethod.invoke(null, ggufFile);

            // Extract model metadata via reflection (nd4j-ggml may not be on compile classpath)
            ModelRegistryEntry.ModelMetadata meta = ModelRegistryEntry.ModelMetadata.builder()
                    .architecture(invokeStringGetter(metadata, "getArchitecture"))
                    .contextLength(invokeIntGetter(metadata, "getContextLength"))
                    .embeddingDimension(invokeIntGetter(metadata, "getHiddenSize"))
                    .numLayers(invokeIntGetter(metadata, "getNumLayers"))
                    .numAttentionHeads(invokeIntGetter(metadata, "getNumAttentionHeads"))
                    .numKVHeads(invokeIntGetter(metadata, "getNumKVHeads"))
                    .feedForwardDimension(invokeIntGetter(metadata, "getIntermediateSize"))
                    .expertCount(invokeIntGetter(metadata, "getExpertCount"))
                    .originalFormat("gguf")
                    .build();

            // Get quantization type from raw metadata
            try {
                Method getRaw = metadata.getClass().getMethod("getRawMetadata");
                Map<String, Object> raw = (Map<String, Object>) getRaw.invoke(metadata);
                Object fileType = raw.get("general.file_type");
                if (fileType instanceof Number) {
                    meta.setQuantizationType(mapGgufFileType(((Number) fileType).intValue()));
                }
            } catch (Exception e) {
                log.debug("Could not extract quantization type: {}", e.getMessage());
            }

            entry.setMetadata(meta);

            // Extract tokenizer info
            try {
                Method getTokenizerInfo = metadata.getClass().getMethod("getTokenizerInfo");
                Object tokInfo = getTokenizerInfo.invoke(metadata);
                if (tokInfo != null) {
                    ModelRegistryEntry.TokenizerConfig tokConfig = ModelRegistryEntry.TokenizerConfig.builder()
                            .bosTokenId(invokeIntGetter(tokInfo, "getBosTokenId"))
                            .eosTokenId(invokeIntGetter(tokInfo, "getEosTokenId"))
                            .padTokenId(invokeIntGetter(tokInfo, "getPadTokenId"))
                            .chatTemplate(invokeStringGetter(tokInfo, "getChatTemplate"))
                            .tokenizerType(invokeStringGetter(tokInfo, "getModel"))
                            .vocabSize(invokeIntGetter(metadata, "getVocabSize"))
                            .build();

                    // Detect chat template format from template content
                    if (tokConfig.getChatTemplate() != null) {
                        tokConfig.setChatTemplateFormat(detectChatTemplateFormat(tokConfig.getChatTemplate()));
                    }

                    // Check for tokenizer.json / vocab file in model directory
                    Path modelDir = ggufFile.getParentFile().toPath();
                    if (Files.exists(modelDir.resolve("tokenizer.json"))) {
                        tokConfig.setVocabFile("tokenizer.json");
                        tokConfig.setTokenizerType("huggingface");
                    } else if (Files.exists(modelDir.resolve("vocab.txt"))) {
                        tokConfig.setVocabFile("vocab.txt");
                    }

                    entry.setTokenizer(tokConfig);
                }
            } catch (Exception e) {
                log.debug("Could not extract tokenizer info: {}", e.getMessage());
            }

            log.info("Extracted GGUF metadata for {}: arch={}, layers={}, ctx={}",
                    entry.getId(), meta.getArchitecture(), meta.getNumLayers(), meta.getContextLength());

        } catch (ClassNotFoundException e) {
            log.warn("nd4j-ggml not on classpath, cannot extract GGUF metadata");
        } catch (Exception e) {
            log.warn("Failed to extract GGUF metadata for {}: {}", entry.getId(), e.getMessage());
        }
    }

    /**
     * Extract just the chat template from a GGUF file without loading the full model.
     */
    private String extractGgufChatTemplate(File ggufFile) {
        try {
            Class<?> importClass = Class.forName("org.nd4j.ggml.GGMLModelImport");
            Method inspectMethod = importClass.getMethod("inspectModel", File.class);
            Object metadata = inspectMethod.invoke(null, ggufFile);

            Method getTokenizerInfo = metadata.getClass().getMethod("getTokenizerInfo");
            Object tokInfo = getTokenizerInfo.invoke(metadata);
            if (tokInfo != null) {
                return invokeStringGetter(tokInfo, "getChatTemplate");
            }
        } catch (Exception e) {
            log.debug("Could not extract chat template from GGUF: {}", e.getMessage());
        }
        return null;
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private ModelRegistryEntry.ModelType inferModelType(String format, Map<String, Object> body) {
        String explicitType = (String) body.get("modelType");
        if (explicitType != null) {
            try {
                return ModelRegistryEntry.ModelType.valueOf(explicitType);
            } catch (IllegalArgumentException ignored) {}
        }
        if ("gguf".equals(format)) return ModelRegistryEntry.ModelType.LLM_GGUF;
        return ModelRegistryEntry.ModelType.OTHER;
    }

    private String detectChatTemplateFormat(String template) {
        if (template == null) return null;
        if (template.contains("<|im_start|>")) return "chatml";
        if (template.contains("[INST]")) return "llama2";
        if (template.contains("### Instruction")) return "alpaca";
        if (template.contains("USER:")) return "vicuna";
        return "unknown";
    }

    /**
     * Map GGUF general.file_type integer to human-readable quantization name.
     * Values from GGML spec.
     */
    private String mapGgufFileType(int fileType) {
        switch (fileType) {
            case 0: return "F32";
            case 1: return "F16";
            case 2: return "Q4_0";
            case 3: return "Q4_1";
            case 7: return "Q8_0";
            case 8: return "Q5_0";
            case 9: return "Q5_1";
            case 10: return "Q2_K";
            case 11: return "Q3_K_S";
            case 12: return "Q3_K_M";
            case 13: return "Q3_K_L";
            case 14: return "Q4_K_S";
            case 15: return "Q4_K_M";
            case 16: return "Q5_K_S";
            case 17: return "Q5_K_M";
            case 18: return "Q6_K";
            case 19: return "IQ2_XXS";
            case 20: return "IQ2_XS";
            case 21: return "IQ3_XXS";
            case 22: return "IQ1_S";
            case 23: return "IQ4_NL";
            case 24: return "IQ3_S";
            case 25: return "IQ2_S";
            case 26: return "IQ4_XS";
            case 28: return "IQ1_M";
            case 29: return "BF16";
            default: return "UNKNOWN(" + fileType + ")";
        }
    }

    private String invokeStringGetter(Object obj, String methodName) {
        try {
            Method m = obj.getClass().getMethod(methodName);
            Object result = m.invoke(obj);
            return result != null ? result.toString() : null;
        } catch (Exception e) {
            return null;
        }
    }

    private int invokeIntGetter(Object obj, String methodName) {
        try {
            Method m = obj.getClass().getMethod(methodName);
            Object result = m.invoke(obj);
            return result instanceof Number ? ((Number) result).intValue() : 0;
        } catch (Exception e) {
            return 0;
        }
    }

    private SameDiff loadOnnxModel(File file) {
        try {
            Class<?> cls = Class.forName("org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter");
            Object importer = cls.getDeclaredConstructor().newInstance();
            Method m = cls.getMethod("runImport", String.class, Map.class, boolean.class, boolean.class);
            return (SameDiff) m.invoke(importer, file.getAbsolutePath(), Collections.emptyMap(), false, false);
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("ONNX import not available");
        } catch (Exception e) {
            throw new RuntimeException("Failed to import ONNX: " + e.getMessage(), e);
        }
    }

    private SameDiff loadGgufModel(File file) {
        try {
            Class<?> cls = Class.forName("org.nd4j.ggml.GGMLModelImport");
            Method m = cls.getMethod("importModel", String.class);
            return (SameDiff) m.invoke(null, file.getAbsolutePath());
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("GGUF import not available");
        } catch (Exception e) {
            throw new RuntimeException("Failed to import GGUF: " + e.getMessage(), e);
        }
    }

    private Map<String, Object> convertGraphInfo(TrainModuleUtils.GraphInfo gi, String modelType) {
        List<Map<String, Object>> nodes = new ArrayList<>();
        List<Map<String, Object>> edges = new ArrayList<>();
        int edgeId = 0;
        for (int i = 0; i < gi.getVertexNames().size(); i++) {
            Map<String, Object> nodeData = new LinkedHashMap<>();
            nodeData.put("id", "layer-" + i);
            nodeData.put("label", gi.getVertexNames().get(i));
            nodeData.put("opType", gi.getVertexTypes().get(i));
            nodeData.put("nodeType", "op");
            Map<String, Object> props = new LinkedHashMap<>();
            if (gi.getVertexInfo().get(i) != null) props.putAll(gi.getVertexInfo().get(i));
            nodeData.put("properties", props);
            Map<String, Object> node = new LinkedHashMap<>();
            node.put("data", nodeData);
            node.put("classes", "op");
            nodes.add(node);
            if (gi.getVertexInputs().get(i) != null) {
                for (int inputIdx : gi.getVertexInputs().get(i)) {
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
        result.put("modelType", modelType);
        result.put("opCount", gi.getVertexNames().size());
        result.put("varCount", 0);
        result.put("paramCount", 0);
        return result;
    }

    private void notifySseClients(String downloadId, DownloadState state) {
        for (RoutingContext client : state.sseClients) {
            try {
                sendSseEvent(client, state);
            } catch (Exception e) {
                state.sseClients.remove(client);
            }
        }
    }

    private void sendSseEvent(RoutingContext rc, DownloadState state) {
        try {
            StringBuilder sb = new StringBuilder();
            sb.append("{\"status\":\"").append(state.status).append("\"");
            sb.append(",\"percent\":").append(state.percent);
            sb.append(",\"fileName\":\"").append(state.fileName).append("\"");
            sb.append(",\"bytesDownloaded\":").append(state.bytesDownloaded);
            sb.append(",\"totalBytes\":").append(state.totalBytes);
            if (state.error != null) {
                sb.append(",\"error\":\"").append(state.error.replace("\"", "'")).append("\"");
            }
            sb.append("}");
            rc.response().write("data: " + sb.toString() + "\n\n");
        } catch (Exception ignored) {
        }
    }

    // ── Cache Management ──────────────────────────────────────────────────

    /**
     * GET /models/cache/status — overview of all caches (DSP, Triton, SDZ).
     */
    private void handleCacheStatus(RoutingContext rc) {
        try {
            Map<String, Object> status = new LinkedHashMap<>();

            // DSP plan disk cache
            Map<String, Object> dsp = new LinkedHashMap<>();
            dsp.put("enabled", DspPlanDiskCache.isEnabled());
            dsp.put("directory", DspPlanDiskCache.getCacheDir());
            dsp.put("overrideDirectory", DspPlanDiskCache.getOverrideDir());
            List<String> dspHashes = DspPlanDiskCache.listCachedHashes();
            dsp.put("entryCount", dspHashes.size());
            dsp.put("totalBytes", computeDirSize(Paths.get(DspPlanDiskCache.getCacheDir())));
            status.put("dsp", dsp);

            // Triton kernel cache
            Map<String, Object> triton = new LinkedHashMap<>();
            boolean tritonAvailable = false;
            try {
                tritonAvailable = TritonCacheManager.isTritonAvailable();
            } catch (Exception ignored) {}
            triton.put("available", tritonAvailable);
            String tritonDir = System.getenv("ND4J_TRITON_CACHE_DIR");
            if (tritonDir == null || tritonDir.isEmpty()) {
                tritonDir = System.getProperty("user.home") + "/.kompile/cache/triton/triton_cache";
            }
            triton.put("directory", tritonDir);
            triton.put("totalBytes", computeDirSize(Paths.get(tritonDir)));
            triton.put("entryCount", countFiles(Paths.get(tritonDir), ".ptx"));
            if (tritonAvailable) {
                try {
                    Method m = Nd4j.getNativeOps().getClass().getMethod("getTritonCacheHitCount");
                    triton.put("cacheHitCount", m.invoke(Nd4j.getNativeOps()));
                } catch (Exception ignored) {}
            }
            status.put("triton", triton);

            // SDZ converted model cache
            Map<String, Object> sdz = new LinkedHashMap<>();
            sdz.put("directory", sdzCacheDir.toString());
            sdz.put("totalBytes", computeDirSize(sdzCacheDir));
            sdz.put("entryCount", countFiles(sdzCacheDir, ".sdz"));
            status.put("sdz", sdz);

            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(status));
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    /**
     * GET /models/cache/dsp — list all cached DSP plan entries with metadata.
     */
    private void handleDspCacheList(RoutingContext rc) {
        try {
            List<String> hashes = DspPlanDiskCache.listCachedHashes();
            String cacheDir = DspPlanDiskCache.getCacheDir();
            List<Map<String, Object>> entries = new ArrayList<>();

            for (String hash : hashes) {
                Map<String, Object> entry = new LinkedHashMap<>();
                entry.put("hash", hash);

                // Read .bin file size
                File binFile = new File(cacheDir, "dsp_" + hash + ".bin");
                if (binFile.exists()) {
                    entry.put("sizeBytes", binFile.length());
                }

                // Read .meta sidecar
                File metaFile = new File(cacheDir, "dsp_" + hash + ".meta");
                if (metaFile.exists()) {
                    Map<String, String> meta = readMetaFile(metaFile);
                    entry.put("metadata", meta);
                }

                entries.add(entry);
            }

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("directory", cacheDir);
            result.put("enabled", DspPlanDiskCache.isEnabled());
            result.put("forceRecompile", DspPlanDiskCache.isForceRecompile());
            result.put("entryCount", entries.size());
            result.put("entries", entries);

            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(result));
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    /**
     * POST /models/cache/dsp/clear — clear all DSP plan cache entries.
     */
    private void handleDspCacheClear(RoutingContext rc) {
        try {
            List<String> hashes = DspPlanDiskCache.listCachedHashes();
            int cleared = 0;
            for (String hash : hashes) {
                try {
                    long structureHash = Long.parseUnsignedLong(hash, 16);
                    DspPlanDiskCache.invalidate(structureHash);
                    cleared++;
                } catch (NumberFormatException e) {
                    log.warn("Invalid DSP cache hash: {}", hash);
                }
            }

            // Also clear model identity index files
            File cacheDir = new File(DspPlanDiskCache.getCacheDir());
            if (cacheDir.isDirectory()) {
                File[] idxFiles = cacheDir.listFiles((d, name) ->
                        name.startsWith("dsp_model_") && name.endsWith(".idx"));
                if (idxFiles != null) {
                    for (File f : idxFiles) {
                        f.delete();
                    }
                }
            }

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("cleared", cleared);
            result.put("directory", DspPlanDiskCache.getCacheDir());
            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(result));
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    /**
     * POST /models/cache/dsp/:hash — invalidate a single DSP plan cache entry.
     */
    private void handleDspCacheInvalidate(RoutingContext rc) {
        String hash = rc.pathParam("hash");
        try {
            long structureHash = Long.parseUnsignedLong(hash, 16);
            boolean existed = DspPlanDiskCache.exists(structureHash);
            DspPlanDiskCache.invalidate(structureHash);

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("hash", hash);
            result.put("existed", existed);
            result.put("invalidated", existed);
            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(result));
        } catch (NumberFormatException e) {
            rc.response().setStatusCode(400).end("{\"error\": \"Invalid hash: " + hash + "\"}");
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    /**
     * GET /models/cache/triton — Triton kernel cache info and listing.
     */
    private void handleTritonCacheInfo(RoutingContext rc) {
        try {
            Map<String, Object> result = new LinkedHashMap<>();

            boolean tritonAvailable = false;
            try {
                tritonAvailable = TritonCacheManager.isTritonAvailable();
            } catch (Exception ignored) {}
            result.put("available", tritonAvailable);

            String tritonDir = System.getenv("ND4J_TRITON_CACHE_DIR");
            if (tritonDir == null || tritonDir.isEmpty()) {
                tritonDir = System.getProperty("user.home") + "/.kompile/cache/triton/triton_cache";
            }
            String overrideDir = System.getenv("ND4J_TRITON_OVERRIDE_DIR");
            if (overrideDir == null || overrideDir.isEmpty()) {
                overrideDir = System.getProperty("user.home") + "/.kompile/cache/triton/triton_override";
            }
            result.put("directory", tritonDir);
            result.put("overrideDirectory", overrideDir);
            result.put("totalBytes", computeDirSize(Paths.get(tritonDir)));

            if (tritonAvailable) {
                try {
                    Method m = Nd4j.getNativeOps().getClass().getMethod("getTritonCacheHitCount");
                    result.put("cacheHitCount", m.invoke(Nd4j.getNativeOps()));
                } catch (Exception ignored) {}
            }

            // List cached kernels
            List<Map<String, Object>> kernels = new ArrayList<>();
            Path tritonPath = Paths.get(tritonDir);
            if (Files.isDirectory(tritonPath)) {
                File[] ptxFiles = tritonPath.toFile().listFiles((d, name) ->
                        name.startsWith("ttir_") && name.endsWith(".ptx"));
                if (ptxFiles != null) {
                    for (File ptx : ptxFiles) {
                        Map<String, Object> kernel = new LinkedHashMap<>();
                        String name = ptx.getName();
                        String kHash = name.substring(5, name.length() - 4); // ttir_<hash>.ptx
                        kernel.put("hash", kHash);
                        kernel.put("sizeBytes", ptx.length());

                        // Read .meta sidecar
                        File metaFile = new File(tritonDir, "ttir_" + kHash + ".meta");
                        if (metaFile.exists()) {
                            kernel.put("metadata", readMetaFile(metaFile));
                        }
                        kernels.add(kernel);
                    }
                }
            }
            result.put("entryCount", kernels.size());
            result.put("entries", kernels);

            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(result));
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    /**
     * POST /models/cache/triton/clear — invalidate all Triton kernel cache.
     */
    private void handleTritonCacheClear(RoutingContext rc) {
        try {
            boolean tritonAvailable = false;
            try {
                tritonAvailable = TritonCacheManager.isTritonAvailable();
            } catch (Exception ignored) {}

            int cleared = 0;

            // Clear in-memory cache via native ops
            if (tritonAvailable) {
                try {
                    Method m = Nd4j.getNativeOps().getClass().getMethod("invalidateTritonCache");
                    m.invoke(Nd4j.getNativeOps());
                } catch (Exception e) {
                    log.warn("Failed to invalidate in-memory Triton cache: {}", e.getMessage());
                }
            }

            // Clear disk cache files
            String tritonDir = System.getenv("ND4J_TRITON_CACHE_DIR");
            if (tritonDir == null || tritonDir.isEmpty()) {
                tritonDir = System.getProperty("user.home") + "/.kompile/cache/triton/triton_cache";
            }
            Path tritonPath = Paths.get(tritonDir);
            if (Files.isDirectory(tritonPath)) {
                File[] files = tritonPath.toFile().listFiles((d, name) ->
                        name.startsWith("ttir_") && (name.endsWith(".ptx") || name.endsWith(".meta")));
                if (files != null) {
                    for (File f : files) {
                        if (f.delete()) cleared++;
                    }
                }
            }

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("cleared", cleared / 2); // ptx + meta pairs
            result.put("filesDeleted", cleared);
            result.put("directory", tritonDir);
            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(result));
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    /**
     * POST /models/cache/triton/export — export Triton cache to a .tkcache bundle.
     * Body: {"outputPath": "/path/to/bundle.tkcache"}
     */
    private void handleTritonCacheExport(RoutingContext rc) {
        try {
            Map<String, Object> body = MAPPER.readValue(rc.getBodyAsString(), Map.class);
            String outputPath = (String) body.get("outputPath");
            if (outputPath == null || outputPath.isEmpty()) {
                // Default to sdzCacheDir parent
                outputPath = sdzCacheDir.getParent().resolve("triton_export.tkcache").toString();
            }

            int exported = TritonCacheManager.exportCache(Path.of(outputPath));

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("exported", exported);
            result.put("outputPath", outputPath);
            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(result));
        } catch (IllegalStateException e) {
            rc.response().setStatusCode(500).end("{\"error\": \"Export failed: " + e.getMessage() + "\"}");
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    /**
     * POST /models/cache/triton/import — import a .tkcache bundle.
     * Body: {"bundlePath": "/path/to/bundle.tkcache", "validateArch": true}
     */
    private void handleTritonCacheImport(RoutingContext rc) {
        try {
            Map<String, Object> body = MAPPER.readValue(rc.getBodyAsString(), Map.class);
            String bundlePath = (String) body.get("bundlePath");
            if (bundlePath == null || bundlePath.isEmpty()) {
                rc.response().setStatusCode(400).end("{\"error\": \"Missing bundlePath\"}");
                return;
            }

            boolean validateArch = body.containsKey("validateArch") ?
                    (Boolean) body.get("validateArch") : true;

            int imported = TritonCacheManager.importCache(Path.of(bundlePath), validateArch);

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("imported", imported);
            result.put("bundlePath", bundlePath);
            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(result));
        } catch (IllegalArgumentException e) {
            rc.response().setStatusCode(400).end("{\"error\": \"" + e.getMessage() + "\"}");
        } catch (IllegalStateException e) {
            rc.response().setStatusCode(500).end("{\"error\": \"Import failed: " + e.getMessage() + "\"}");
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    /**
     * GET /models/cache/sdz — list cached SDZ model files.
     */
    private void handleSdzCacheList(RoutingContext rc) {
        try {
            List<Map<String, Object>> entries = new ArrayList<>();

            if (Files.isDirectory(sdzCacheDir)) {
                File[] sdzFiles = sdzCacheDir.toFile().listFiles((d, name) -> name.endsWith(".sdz"));
                if (sdzFiles != null) {
                    for (File f : sdzFiles) {
                        Map<String, Object> entry = new LinkedHashMap<>();
                        entry.put("filename", f.getName());
                        entry.put("sizeBytes", f.length());
                        entry.put("lastModified", f.lastModified());
                        entry.put("path", f.getAbsolutePath());
                        entries.add(entry);
                    }
                }
            }

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("directory", sdzCacheDir.toString());
            result.put("entryCount", entries.size());
            result.put("totalBytes", entries.stream().mapToLong(e -> (Long) e.get("sizeBytes")).sum());
            result.put("entries", entries);

            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(result));
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    /**
     * POST /models/cache/sdz/clear — clear all cached SDZ model files.
     */
    private void handleSdzCacheClear(RoutingContext rc) {
        try {
            int cleared = 0;
            if (Files.isDirectory(sdzCacheDir)) {
                File[] sdzFiles = sdzCacheDir.toFile().listFiles((d, name) -> name.endsWith(".sdz"));
                if (sdzFiles != null) {
                    for (File f : sdzFiles) {
                        if (f.delete()) cleared++;
                    }
                }
            }

            // Clear sdzPath references in registry entries
            for (ModelRegistryEntry entry : registry.listEntries()) {
                if (entry.isSdzCached()) {
                    entry.setSdzCached(false);
                    entry.setSdzPath(null);
                    registry.addEntry(entry);
                }
            }

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("cleared", cleared);
            result.put("directory", sdzCacheDir.toString());
            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(result));
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    /**
     * POST /models/cache/sdz/:filename — evict a single SDZ cache file.
     */
    private void handleSdzCacheEvict(RoutingContext rc) {
        String filename = rc.pathParam("filename");
        try {
            // Sanitize: only allow .sdz files within the cache directory
            if (!filename.endsWith(".sdz") || filename.contains("..") || filename.contains("/")) {
                rc.response().setStatusCode(400).end("{\"error\": \"Invalid filename\"}");
                return;
            }

            Path target = sdzCacheDir.resolve(filename);
            boolean existed = Files.exists(target);
            if (existed) {
                Files.delete(target);

                // Clear sdzPath in any registry entry pointing to this file
                String targetPath = target.toAbsolutePath().toString();
                for (ModelRegistryEntry entry : registry.listEntries()) {
                    if (targetPath.equals(entry.getSdzPath())) {
                        entry.setSdzCached(false);
                        entry.setSdzPath(null);
                        registry.addEntry(entry);
                    }
                }
            }

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("filename", filename);
            result.put("existed", existed);
            result.put("deleted", existed);
            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(MAPPER.writeValueAsString(result));
        } catch (Exception e) {
            rc.response().setStatusCode(500).end("{\"error\": \"" + e.getMessage() + "\"}");
        }
    }

    // ── Cache utility methods ──────────────────────────────────────────────

    private long computeDirSize(Path dir) {
        if (!Files.isDirectory(dir)) return 0;
        try {
            return Files.walk(dir)
                    .filter(Files::isRegularFile)
                    .mapToLong(p -> {
                        try { return Files.size(p); }
                        catch (Exception e) { return 0; }
                    })
                    .sum();
        } catch (Exception e) {
            return 0;
        }
    }

    private int countFiles(Path dir, String suffix) {
        if (!Files.isDirectory(dir)) return 0;
        File[] files = dir.toFile().listFiles((d, name) -> name.endsWith(suffix));
        return files != null ? files.length : 0;
    }

    private Map<String, String> readMetaFile(File metaFile) {
        Map<String, String> meta = new LinkedHashMap<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(metaFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                int eq = line.indexOf('=');
                if (eq > 0) {
                    meta.put(line.substring(0, eq), line.substring(eq + 1));
                }
            }
        } catch (Exception ignored) {}
        return meta;
    }

    // ── UIModule interface ─────────────────────────────────────────────────

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
