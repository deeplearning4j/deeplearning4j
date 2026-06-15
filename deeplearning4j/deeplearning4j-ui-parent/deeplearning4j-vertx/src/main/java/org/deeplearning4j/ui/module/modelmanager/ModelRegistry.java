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

package org.deeplearning4j.ui.module.modelmanager;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.shade.jackson.core.type.TypeReference;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Manages a local registry of downloaded models.
 * Persists as registry.json under ~/.deeplearning4j/models/.
 * Uses atomic writes (temp file + rename) and backup on each save.
 */
@Slf4j
public class ModelRegistry {

    private static final ObjectMapper MAPPER = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

    private final Path registryDir;
    private final Path registryFile;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    private Map<String, ModelRegistryEntry> entries = new LinkedHashMap<>();
    private long lastModified = 0;

    public ModelRegistry() {
        this(Paths.get(System.getProperty("user.home"), ".deeplearning4j", "models"));
    }

    public ModelRegistry(Path registryDir) {
        this.registryDir = registryDir;
        this.registryFile = registryDir.resolve("registry.json");
        try {
            Files.createDirectories(registryDir);
        } catch (IOException e) {
            log.warn("Could not create registry directory: {}", registryDir, e);
        }
        loadRegistry();
    }

    public Path getRegistryDir() {
        return registryDir;
    }

    public List<ModelRegistryEntry> listEntries() {
        refreshIfStale();
        lock.readLock().lock();
        try {
            return new ArrayList<>(entries.values());
        } finally {
            lock.readLock().unlock();
        }
    }

    public ModelRegistryEntry getEntry(String id) {
        refreshIfStale();
        lock.readLock().lock();
        try {
            return entries.get(id);
        } finally {
            lock.readLock().unlock();
        }
    }

    public void addEntry(ModelRegistryEntry entry) {
        lock.writeLock().lock();
        try {
            entries.put(entry.getId(), entry);
            saveRegistry();
        } finally {
            lock.writeLock().unlock();
        }
    }

    public ModelRegistryEntry removeEntry(String id) {
        lock.writeLock().lock();
        try {
            ModelRegistryEntry removed = entries.remove(id);
            if (removed != null) {
                saveRegistry();
            }
            return removed;
        } finally {
            lock.writeLock().unlock();
        }
    }

    public void updateStatus(String id, ModelRegistryEntry.ModelStatus status) {
        lock.writeLock().lock();
        try {
            ModelRegistryEntry entry = entries.get(id);
            if (entry != null) {
                entry.setStatus(status);
                saveRegistry();
            }
        } finally {
            lock.writeLock().unlock();
        }
    }

    private void loadRegistry() {
        if (!Files.exists(registryFile)) {
            entries = new LinkedHashMap<>();
            return;
        }
        try {
            List<ModelRegistryEntry> list = MAPPER.readValue(registryFile.toFile(),
                    new TypeReference<List<ModelRegistryEntry>>() {});
            entries = new LinkedHashMap<>();
            for (ModelRegistryEntry e : list) {
                entries.put(e.getId(), e);
            }
            lastModified = Files.getLastModifiedTime(registryFile).toMillis();
        } catch (IOException e) {
            log.error("Failed to load registry from {}", registryFile, e);
            entries = new LinkedHashMap<>();
        }
    }

    private void saveRegistry() {
        try {
            // Backup existing
            if (Files.exists(registryFile)) {
                Path backup = registryDir.resolve("registry.json.bak");
                Files.copy(registryFile, backup, StandardCopyOption.REPLACE_EXISTING);
            }
            // Atomic write
            Path tmp = registryDir.resolve("registry.json.tmp");
            MAPPER.writeValue(tmp.toFile(), new ArrayList<>(entries.values()));
            Files.move(tmp, registryFile, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            lastModified = Files.getLastModifiedTime(registryFile).toMillis();
        } catch (IOException e) {
            log.error("Failed to save registry to {}", registryFile, e);
        }
    }

    private void refreshIfStale() {
        try {
            if (Files.exists(registryFile)) {
                long currentMod = Files.getLastModifiedTime(registryFile).toMillis();
                if (currentMod > lastModified) {
                    lock.writeLock().lock();
                    try {
                        // Double-check after acquiring write lock
                        currentMod = Files.getLastModifiedTime(registryFile).toMillis();
                        if (currentMod > lastModified) {
                            loadRegistry();
                        }
                    } finally {
                        lock.writeLock().unlock();
                    }
                }
            }
        } catch (IOException e) {
            // ignore
        }
    }
}
