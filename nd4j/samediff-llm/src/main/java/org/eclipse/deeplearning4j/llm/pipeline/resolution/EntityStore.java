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

package org.eclipse.deeplearning4j.llm.pipeline.resolution;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * In-memory entity store that performs resolution, deduplication,
 * and clustering of entity mentions. Supports manual overrides
 * (merge, split, rename, retype) from the UI.
 *
 * <p>Thread-safe for concurrent ingestion from multiple pipeline stages.</p>
 */
public class EntityStore {

    private final Map<String, ResolvedEntity> entitiesById = new ConcurrentHashMap<>();
    private final Map<String, EntityResolutionStrategy> strategies = new ConcurrentHashMap<>();
    private final Map<String, Double> thresholdOverrides = new ConcurrentHashMap<>();
    private final AtomicLong idCounter = new AtomicLong(1);

    public EntityStore() {
        registerStrategy(new PersonResolutionStrategy());
        registerStrategy(new OrganizationResolutionStrategy());
        registerStrategy(new LocationResolutionStrategy());
    }

    public void registerStrategy(EntityResolutionStrategy strategy) {
        strategies.put(strategy.entityType().toUpperCase(), strategy);
    }

    /**
     * Set a custom threshold for a specific entity type.
     */
    public void setThreshold(String entityType, double threshold) {
        thresholdOverrides.put(entityType.toUpperCase(), threshold);
    }

    /**
     * Ingest a new entity mention. Resolves it against existing entities
     * using the appropriate type-specific strategy. If no match is found
     * above threshold, creates a new resolved entity.
     *
     * @param text           surface form text
     * @param type           entity type (PERSON, ORGANIZATION, etc.)
     * @param sourceDocument source document identifier
     * @return the resolved entity this mention was assigned to
     */
    public synchronized ResolvedEntity ingest(String text, String type, String sourceDocument) {
        String upperType = type.toUpperCase();
        EntityResolutionStrategy strategy = getStrategyForType(upperType);
        double threshold = thresholdOverrides.getOrDefault(upperType, strategy.defaultThreshold());

        ResolvedEntity bestMatch = null;
        double bestScore = 0.0;

        for (ResolvedEntity existing : entitiesById.values()) {
            if (!existing.getType().equalsIgnoreCase(upperType)) continue;

            // Compare against canonical name and all surface forms
            double score = strategy.computeSimilarity(text, existing.getCanonicalName());
            for (String surfaceForm : existing.surfaceForms()) {
                double formScore = strategy.computeSimilarity(text, surfaceForm);
                score = Math.max(score, formScore);
            }

            if (score > bestScore) {
                bestScore = score;
                bestMatch = existing;
            }
        }

        EntityMention mention = new EntityMention(text, upperType, sourceDocument, bestScore);

        if (bestMatch != null && bestScore >= threshold) {
            mention.setConfidence(bestScore);
            bestMatch.addMention(mention);
            return bestMatch;
        }

        // No match — create new entity
        String id = "E" + idCounter.getAndIncrement();
        ResolvedEntity newEntity = new ResolvedEntity(id, text, upperType);
        mention.setConfidence(1.0); // self-match is certain
        newEntity.addMention(mention);
        entitiesById.put(id, newEntity);
        return newEntity;
    }

    /**
     * Ingest a batch of entities from a pipeline result.
     */
    public List<ResolvedEntity> ingestBatch(List<? extends Object> entities, String sourceDocument) {
        List<ResolvedEntity> results = new ArrayList<>();
        for (Object entity : entities) {
            if (entity instanceof org.eclipse.deeplearning4j.llm.pipeline.ModelType.Entity) {
                org.eclipse.deeplearning4j.llm.pipeline.ModelType.Entity e =
                        (org.eclipse.deeplearning4j.llm.pipeline.ModelType.Entity) entity;
                results.add(ingest(e.getText(), e.getType(), sourceDocument));
            }
        }
        return results;
    }

    // --- Query methods ---

    public ResolvedEntity getById(String id) {
        return entitiesById.get(id);
    }

    public Collection<ResolvedEntity> getAllEntities() {
        return Collections.unmodifiableCollection(entitiesById.values());
    }

    public List<ResolvedEntity> getByType(String type) {
        List<ResolvedEntity> result = new ArrayList<>();
        for (ResolvedEntity e : entitiesById.values()) {
            if (e.getType().equalsIgnoreCase(type)) result.add(e);
        }
        return result;
    }

    public int entityCount() {
        return entitiesById.size();
    }

    /**
     * Find potential duplicates that fall between the threshold and a lower bound.
     * These are candidates for manual review.
     */
    public List<ResolutionCandidate> findCandidates(double minConfidence) {
        List<ResolutionCandidate> candidates = new ArrayList<>();
        List<ResolvedEntity> all = new ArrayList<>(entitiesById.values());

        for (int i = 0; i < all.size(); i++) {
            for (int j = i + 1; j < all.size(); j++) {
                ResolvedEntity a = all.get(i);
                ResolvedEntity b = all.get(j);
                if (!a.getType().equals(b.getType())) continue;

                EntityResolutionStrategy strategy = getStrategyForType(a.getType());
                double score = strategy.computeSimilarity(a.getCanonicalName(), b.getCanonicalName());

                // Also check all surface forms
                for (String fa : a.surfaceForms()) {
                    for (String fb : b.surfaceForms()) {
                        score = Math.max(score, strategy.computeSimilarity(fa, fb));
                    }
                }

                if (score >= minConfidence) {
                    candidates.add(new ResolutionCandidate(a, b, score));
                }
            }
        }

        candidates.sort((x, y) -> Double.compare(y.getConfidence(), x.getConfidence()));
        return candidates;
    }

    // --- Edit operations (from UI) ---

    /**
     * Merge two entities into one. All mentions from entityB move to entityA.
     */
    public synchronized ResolvedEntity merge(String entityIdA, String entityIdB) {
        ResolvedEntity a = entitiesById.get(entityIdA);
        ResolvedEntity b = entitiesById.get(entityIdB);
        if (a == null || b == null) throw new IllegalArgumentException(
                "Entity not found: " + (a == null ? entityIdA : entityIdB));

        for (EntityMention mention : b.getMentions()) {
            a.addMention(mention);
        }
        a.setManuallyVerified(true);
        entitiesById.remove(entityIdB);
        return a;
    }

    /**
     * Split a mention out of an entity into its own new entity.
     */
    public synchronized ResolvedEntity split(String entityId, int mentionIndex) {
        ResolvedEntity source = entitiesById.get(entityId);
        if (source == null) throw new IllegalArgumentException("Entity not found: " + entityId);
        if (mentionIndex < 0 || mentionIndex >= source.getMentions().size())
            throw new IllegalArgumentException("Invalid mention index: " + mentionIndex);

        EntityMention mention = source.getMentions().remove(mentionIndex);
        String newId = "E" + idCounter.getAndIncrement();
        ResolvedEntity newEntity = new ResolvedEntity(newId, mention.getText(), mention.getType());
        mention.setConfidence(1.0);
        newEntity.addMention(mention);
        newEntity.setManuallyVerified(true);
        entitiesById.put(newId, newEntity);

        if (source.getMentions().isEmpty()) {
            entitiesById.remove(entityId);
        }

        return newEntity;
    }

    /**
     * Update the canonical name of an entity.
     */
    public synchronized void rename(String entityId, String newCanonicalName) {
        ResolvedEntity entity = entitiesById.get(entityId);
        if (entity == null) throw new IllegalArgumentException("Entity not found: " + entityId);
        entity.setCanonicalName(newCanonicalName);
        entity.setManuallyVerified(true);
    }

    /**
     * Change the type of an entity.
     */
    public synchronized void retype(String entityId, String newType) {
        ResolvedEntity entity = entitiesById.get(entityId);
        if (entity == null) throw new IllegalArgumentException("Entity not found: " + entityId);
        entity.setType(newType.toUpperCase());
        entity.setManuallyVerified(true);
    }

    /**
     * Delete an entity entirely.
     */
    public synchronized void delete(String entityId) {
        entitiesById.remove(entityId);
    }

    /**
     * Clear all entities.
     */
    public synchronized void clear() {
        entitiesById.clear();
        idCounter.set(1);
    }

    /**
     * Get statistics about the store.
     */
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("totalEntities", entitiesById.size());

        Map<String, Integer> byType = new LinkedHashMap<>();
        int totalMentions = 0;
        int verified = 0;
        for (ResolvedEntity e : entitiesById.values()) {
            byType.merge(e.getType(), 1, Integer::sum);
            totalMentions += e.mentionCount();
            if (e.isManuallyVerified()) verified++;
        }
        stats.put("totalMentions", totalMentions);
        stats.put("verifiedEntities", verified);
        stats.put("byType", byType);

        return stats;
    }

    private EntityResolutionStrategy getStrategyForType(String type) {
        EntityResolutionStrategy strategy = strategies.get(type.toUpperCase());
        if (strategy == null) {
            strategy = new DefaultResolutionStrategy(type.toUpperCase());
            strategies.put(type.toUpperCase(), strategy);
        }
        return strategy;
    }

    /**
     * A candidate pair of entities that might be the same.
     */
    public static class ResolutionCandidate {
        @lombok.Getter
        private final ResolvedEntity entityA;
        @lombok.Getter
        private final ResolvedEntity entityB;
        @lombok.Getter
        private final double confidence;

        public ResolutionCandidate(ResolvedEntity entityA, ResolvedEntity entityB, double confidence) {
            this.entityA = entityA;
            this.entityB = entityB;
            this.confidence = confidence;
        }
    }
}
