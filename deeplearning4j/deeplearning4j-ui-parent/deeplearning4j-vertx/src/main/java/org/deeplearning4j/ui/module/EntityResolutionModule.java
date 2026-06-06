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

import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.RoutingContext;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.core.storage.StatsStorageEvent;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.i18n.I18NResource;
import org.eclipse.deeplearning4j.llm.pipeline.resolution.*;

import java.util.*;
import java.util.function.BiConsumer;

/**
 * UI module for entity resolution. Provides a web interface to:
 * <ul>
 *   <li>View all resolved entities with confidence scores</li>
 *   <li>See entity clusters and surface-form mentions</li>
 *   <li>Manually merge, split, rename, and retype entities</li>
 *   <li>Review resolution candidates (potential duplicates)</li>
 *   <li>Ingest new entities for resolution</li>
 *   <li>Configure per-type resolution thresholds</li>
 * </ul>
 */
@Slf4j
public class EntityResolutionModule implements UIModule {

    private final EntityStore entityStore;

    public EntityResolutionModule() {
        this.entityStore = new EntityStore();
    }

    public EntityResolutionModule(EntityStore entityStore) {
        this.entityStore = entityStore;
    }

    public EntityStore getEntityStore() {
        return entityStore;
    }

    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.emptyList();
    }

    @Override
    public List<Route> getRoutes() {
        List<Route> routes = new ArrayList<>();

        // HTML page
        routes.add(new Route("/entities", HttpMethod.GET, (path, rc) ->
                rc.response().sendFile("templates/EntityResolution.html")));

        // REST API - entities
        routes.add(new Route("/api/entities", HttpMethod.GET, (path, rc) -> handleGetEntities(rc)));
        routes.add(new Route("/api/entities/stats", HttpMethod.GET, (path, rc) -> handleGetStats(rc)));
        routes.add(new Route("/api/entities/candidates", HttpMethod.GET, (path, rc) -> handleGetCandidates(rc)));
        routes.add(new Route("/api/entities/ingest", HttpMethod.POST, (path, rc) -> handleIngest(rc)));
        routes.add(new Route("/api/entities/ingestBatch", HttpMethod.POST, (path, rc) -> handleIngestBatch(rc)));
        routes.add(new Route("/api/entities/:entityId/rename", HttpMethod.POST, (path, rc) -> handleRename(rc)));
        routes.add(new Route("/api/entities/:entityId/retype", HttpMethod.POST, (path, rc) -> handleRetype(rc)));
        routes.add(new Route("/api/entities/:entityId/split", HttpMethod.POST, (path, rc) -> handleSplit(rc)));
        routes.add(new Route("/api/entities/merge", HttpMethod.POST, (path, rc) -> handleMerge(rc)));
        routes.add(new Route("/api/entities/:entityId", HttpMethod.DELETE, (path, rc) -> handleDelete(rc)));
        routes.add(new Route("/api/entities/threshold", HttpMethod.POST, (path, rc) -> handleSetThreshold(rc)));
        routes.add(new Route("/api/entities/clear", HttpMethod.POST, (path, rc) -> handleClear(rc)));

        return routes;
    }

    private void handleGetEntities(RoutingContext rc) {
        try {
            String typeFilter = rc.request().getParam("type");
            String sortBy = Optional.ofNullable(rc.request().getParam("sort")).orElse("confidence");
            String order = Optional.ofNullable(rc.request().getParam("order")).orElse("desc");

            Collection<ResolvedEntity> entities;
            if (typeFilter != null && !typeFilter.isEmpty()) {
                entities = entityStore.getByType(typeFilter);
            } else {
                entities = entityStore.getAllEntities();
            }

            List<ResolvedEntity> sorted = new ArrayList<>(entities);
            Comparator<ResolvedEntity> cmp;
            switch (sortBy) {
                case "name":
                    cmp = Comparator.comparing(e -> e.getCanonicalName().toLowerCase());
                    break;
                case "type":
                    cmp = Comparator.comparing(ResolvedEntity::getType);
                    break;
                case "mentions":
                    cmp = Comparator.comparingInt(ResolvedEntity::mentionCount);
                    break;
                default:
                    cmp = Comparator.comparingDouble(ResolvedEntity::averageConfidence);
                    break;
            }
            if ("desc".equals(order)) cmp = cmp.reversed();
            sorted.sort(cmp);

            JsonArray arr = new JsonArray();
            for (ResolvedEntity e : sorted) {
                arr.add(entityToJson(e));
            }

            rc.response()
                    .putHeader("Content-Type", "application/json")
                    .end(arr.encode());
        } catch (Exception e) {
            errorResponse(rc, 500, e.getMessage());
        }
    }

    private void handleGetStats(RoutingContext rc) {
        try {
            Map<String, Object> stats = entityStore.getStats();
            rc.response()
                    .putHeader("Content-Type", "application/json")
                    .end(new JsonObject(stats).encode());
        } catch (Exception e) {
            errorResponse(rc, 500, e.getMessage());
        }
    }

    private void handleGetCandidates(RoutingContext rc) {
        try {
            double minConf = Double.parseDouble(Optional.ofNullable(rc.request().getParam("minConfidence")).orElse("0.4"));
            List<EntityStore.ResolutionCandidate> candidates = entityStore.findCandidates(minConf);

            JsonArray arr = new JsonArray();
            for (EntityStore.ResolutionCandidate c : candidates) {
                JsonObject obj = new JsonObject();
                obj.put("entityA", entityToJson(c.getEntityA()));
                obj.put("entityB", entityToJson(c.getEntityB()));
                obj.put("confidence", Math.round(c.getConfidence() * 1000.0) / 1000.0);
                arr.add(obj);
            }

            rc.response()
                    .putHeader("Content-Type", "application/json")
                    .end(arr.encode());
        } catch (Exception e) {
            errorResponse(rc, 500, e.getMessage());
        }
    }

    private void handleIngest(RoutingContext rc) {
        try {
            JsonObject json = rc.getBodyAsJson();
            String text = json.getString("text");
            String type = json.getString("type");
            String source = json.getString("source", "manual");

            if (text == null || type == null) {
                errorResponse(rc, 400, "Missing required fields: text, type");
                return;
            }

            ResolvedEntity result = entityStore.ingest(text, type, source);
            rc.response()
                    .putHeader("Content-Type", "application/json")
                    .end(entityToJson(result).encode());
        } catch (Exception e) {
            errorResponse(rc, 500, e.getMessage());
        }
    }

    private void handleIngestBatch(RoutingContext rc) {
        try {
            JsonObject json = rc.getBodyAsJson();
            JsonArray entities = json.getJsonArray("entities");
            String source = json.getString("source", "batch");

            JsonArray results = new JsonArray();
            for (int i = 0; i < entities.size(); i++) {
                JsonObject ent = entities.getJsonObject(i);
                ResolvedEntity result = entityStore.ingest(
                        ent.getString("text"),
                        ent.getString("type"),
                        source);
                results.add(entityToJson(result));
            }

            rc.response()
                    .putHeader("Content-Type", "application/json")
                    .end(results.encode());
        } catch (Exception e) {
            errorResponse(rc, 500, e.getMessage());
        }
    }

    private void handleRename(RoutingContext rc) {
        try {
            String entityId = rc.request().getParam("entityId");
            JsonObject json = rc.getBodyAsJson();
            String newName = json.getString("name");
            if (newName == null) {
                errorResponse(rc, 400, "Missing field: name");
                return;
            }
            entityStore.rename(entityId, newName);
            ResolvedEntity entity = entityStore.getById(entityId);
            rc.response()
                    .putHeader("Content-Type", "application/json")
                    .end(entityToJson(entity).encode());
        } catch (Exception e) {
            errorResponse(rc, 400, e.getMessage());
        }
    }

    private void handleRetype(RoutingContext rc) {
        try {
            String entityId = rc.request().getParam("entityId");
            JsonObject json = rc.getBodyAsJson();
            String newType = json.getString("type");
            if (newType == null) {
                errorResponse(rc, 400, "Missing field: type");
                return;
            }
            entityStore.retype(entityId, newType);
            ResolvedEntity entity = entityStore.getById(entityId);
            rc.response()
                    .putHeader("Content-Type", "application/json")
                    .end(entityToJson(entity).encode());
        } catch (Exception e) {
            errorResponse(rc, 400, e.getMessage());
        }
    }

    private void handleSplit(RoutingContext rc) {
        try {
            String entityId = rc.request().getParam("entityId");
            JsonObject json = rc.getBodyAsJson();
            int mentionIndex = json.getInteger("mentionIndex", 0);
            ResolvedEntity newEntity = entityStore.split(entityId, mentionIndex);
            rc.response()
                    .putHeader("Content-Type", "application/json")
                    .end(entityToJson(newEntity).encode());
        } catch (Exception e) {
            errorResponse(rc, 400, e.getMessage());
        }
    }

    private void handleMerge(RoutingContext rc) {
        try {
            JsonObject json = rc.getBodyAsJson();
            String idA = json.getString("entityIdA");
            String idB = json.getString("entityIdB");
            if (idA == null || idB == null) {
                errorResponse(rc, 400, "Missing fields: entityIdA, entityIdB");
                return;
            }
            ResolvedEntity merged = entityStore.merge(idA, idB);
            rc.response()
                    .putHeader("Content-Type", "application/json")
                    .end(entityToJson(merged).encode());
        } catch (Exception e) {
            errorResponse(rc, 400, e.getMessage());
        }
    }

    private void handleDelete(RoutingContext rc) {
        try {
            String entityId = rc.request().getParam("entityId");
            entityStore.delete(entityId);
            rc.response()
                    .putHeader("Content-Type", "application/json")
                    .end(new JsonObject().put("deleted", entityId).encode());
        } catch (Exception e) {
            errorResponse(rc, 500, e.getMessage());
        }
    }

    private void handleSetThreshold(RoutingContext rc) {
        try {
            JsonObject json = rc.getBodyAsJson();
            String type = json.getString("type");
            double threshold = json.getDouble("threshold");
            if (type == null) {
                errorResponse(rc, 400, "Missing field: type");
                return;
            }
            entityStore.setThreshold(type, threshold);
            rc.response()
                    .putHeader("Content-Type", "application/json")
                    .end(new JsonObject()
                            .put("type", type)
                            .put("threshold", threshold).encode());
        } catch (Exception e) {
            errorResponse(rc, 400, e.getMessage());
        }
    }

    private void handleClear(RoutingContext rc) {
        entityStore.clear();
        rc.response()
                .putHeader("Content-Type", "application/json")
                .end(new JsonObject().put("cleared", true).encode());
    }

    private JsonObject entityToJson(ResolvedEntity e) {
        JsonObject obj = new JsonObject();
        obj.put("id", e.getId());
        obj.put("canonicalName", e.getCanonicalName());
        obj.put("type", e.getType());
        obj.put("mentionCount", e.mentionCount());
        obj.put("averageConfidence", Math.round(e.averageConfidence() * 1000.0) / 1000.0);
        obj.put("maxConfidence", Math.round(e.maxConfidence() * 1000.0) / 1000.0);
        obj.put("manuallyVerified", e.isManuallyVerified());

        JsonArray surfaceForms = new JsonArray();
        for (String form : e.surfaceForms()) {
            surfaceForms.add(form);
        }
        obj.put("surfaceForms", surfaceForms);

        JsonArray mentions = new JsonArray();
        for (EntityMention m : e.getMentions()) {
            JsonObject mObj = new JsonObject();
            mObj.put("text", m.getText());
            mObj.put("type", m.getType());
            mObj.put("source", m.getSourceDocument());
            mObj.put("confidence", Math.round(m.getConfidence() * 1000.0) / 1000.0);
            mentions.add(mObj);
        }
        obj.put("mentions", mentions);

        return obj;
    }

    private void errorResponse(RoutingContext rc, int status, String message) {
        rc.response()
                .setStatusCode(status)
                .putHeader("Content-Type", "application/json")
                .end(new JsonObject().put("error", message).encode());
    }

    @Override
    public void reportStorageEvents(Collection<StatsStorageEvent> events) {
        // Not used — entity data comes via REST API, not StatsStorage
    }

    @Override
    public void onAttach(StatsStorage statsStorage) {}

    @Override
    public void onDetach(StatsStorage statsStorage) {}

    @Override
    public List<I18NResource> getInternationalizationResources() {
        return Collections.emptyList();
    }
}
