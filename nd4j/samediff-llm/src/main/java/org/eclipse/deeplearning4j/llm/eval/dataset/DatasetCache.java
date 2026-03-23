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

package org.eclipse.deeplearning4j.llm.eval.dataset;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.model.download.ModelDownloader;

import java.io.File;
import java.io.IOException;

/**
 * File-based dataset caching utility.
 * Reuses the ModelDownloader infrastructure for downloading and caching dataset files.
 */
@Slf4j
public class DatasetCache {

    private static final String CACHE_DIR_PROPERTY = "llm.eval.cache.dir";
    private static final String DEFAULT_CACHE_DIR = System.getProperty("user.home") + "/.cache/dl4j-eval-datasets";

    public static File getCacheDir() {
        return ModelDownloader.getCacheDir(CACHE_DIR_PROPERTY, DEFAULT_CACHE_DIR);
    }

    public static File downloadIfNeeded(String url, String fileName) throws IOException {
        File cacheDir = getCacheDir();
        ModelDownloader.DownloadResult result = ModelDownloader.download(url, fileName, cacheDir);
        if (result.isDownloadedNow()) {
            log.info("Downloaded dataset file {} ({} bytes)", fileName, result.getFileSizeBytes());
        } else {
            log.debug("Using cached dataset file {}", fileName);
        }
        return result.getModelFile();
    }

    public static boolean isCached(String fileName) {
        return ModelDownloader.isCached(fileName, getCacheDir());
    }

    public static void clearCache() throws IOException {
        ModelDownloader.clearCache(getCacheDir());
        log.info("Cleared dataset cache at {}", getCacheDir());
    }
}
