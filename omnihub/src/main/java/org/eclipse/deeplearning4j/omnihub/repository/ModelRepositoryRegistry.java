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
package org.eclipse.deeplearning4j.omnihub.repository;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Registry of {@link ModelRepository} instances with priority-based fallback.
 * On resolve, tries each repository in priority order (lowest first) until one succeeds.
 *
 * @author Adam Gibson
 */
public class ModelRepositoryRegistry {

    private static final Logger log = LoggerFactory.getLogger(ModelRepositoryRegistry.class);

    private static volatile ModelRepositoryRegistry defaultInstance;

    private final List<ModelRepository> repositories = new ArrayList<>();

    /**
     * Returns the default singleton registry with HuggingFace (priority=10) and GitHub (priority=20).
     */
    public static ModelRepositoryRegistry getDefault() {
        if (defaultInstance == null) {
            synchronized (ModelRepositoryRegistry.class) {
                if (defaultInstance == null) {
                    ModelRepositoryRegistry registry = new ModelRepositoryRegistry();
                    registry.addRepository(new HuggingFaceModelRepository());
                    registry.addRepository(new GitHubModelRepository());
                    defaultInstance = registry;
                }
            }
        }
        return defaultInstance;
    }

    /**
     * Replaces the default singleton registry.
     */
    public static void setDefault(ModelRepositoryRegistry registry) {
        synchronized (ModelRepositoryRegistry.class) {
            defaultInstance = registry;
        }
    }

    /**
     * Adds a repository and re-sorts by priority.
     */
    public void addRepository(ModelRepository repo) {
        repositories.add(repo);
        repositories.sort(Comparator.comparingInt(ModelRepository::priority));
    }

    /**
     * Removes a repository by name.
     */
    public void removeRepository(String name) {
        repositories.removeIf(r -> r.name().equals(name));
    }

    /**
     * Returns the repository with the given name, or null if not found.
     */
    public ModelRepository getRepository(String name) {
        for (ModelRepository repo : repositories) {
            if (repo.name().equals(name)) {
                return repo;
            }
        }
        return null;
    }

    /**
     * Returns an unmodifiable view of all registered repositories, sorted by priority.
     */
    public List<ModelRepository> getRepositories() {
        return List.copyOf(repositories);
    }

    /**
     * Resolves a model by trying each repository in priority order.
     * Returns the local file from the first repository that succeeds.
     *
     * @throws IOException if no repository can resolve the model
     */
    public File resolve(String modelName, String framework, boolean forceDownload) throws IOException {
        IOException lastException = null;
        for (ModelRepository repo : repositories) {
            if (!repo.canResolve(modelName, framework)) {
                continue;
            }
            try {
                File result = repo.resolve(modelName, framework, forceDownload);
                log.debug("Resolved {} from repository '{}'", modelName, repo.name());
                return result;
            } catch (IOException e) {
                log.debug("Repository '{}' failed to resolve {}: {}", repo.name(), modelName, e.getMessage());
                lastException = e;
            }
        }
        throw lastException != null ? lastException
                : new IOException("No repository can resolve model: " + modelName + " (framework: " + framework + ")");
    }

    /**
     * Resolves a HuggingFace model by trying each HF-capable repository in priority order.
     *
     * @throws IOException if no repository can resolve the model
     */
    public File resolveHuggingFace(String huggingFaceId, String filePattern, boolean forceDownload) throws IOException {
        IOException lastException = null;
        for (ModelRepository repo : repositories) {
            if (!repo.supportsHuggingFace()) {
                continue;
            }
            try {
                File result = repo.resolveHuggingFace(huggingFaceId, filePattern, forceDownload);
                log.debug("Resolved HuggingFace {} from repository '{}'", huggingFaceId, repo.name());
                return result;
            } catch (IOException e) {
                log.debug("Repository '{}' failed to resolve HuggingFace {}: {}", repo.name(), huggingFaceId, e.getMessage());
                lastException = e;
            }
        }
        throw lastException != null ? lastException
                : new IOException("No repository can resolve HuggingFace model: " + huggingFaceId);
    }

    /**
     * Resolves a model from a specific named repository only.
     *
     * @throws IOException if the named repository cannot resolve the model or doesn't exist
     */
    public File resolveFromRepository(String repositoryName, String modelName, String framework, boolean forceDownload) throws IOException {
        ModelRepository repo = getRepository(repositoryName);
        if (repo == null) {
            throw new IOException("No repository found with name: " + repositoryName);
        }
        return repo.resolve(modelName, framework, forceDownload);
    }
}
