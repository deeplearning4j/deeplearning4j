/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.datasets.fetchers;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.nd4j.util.ArchiveUtils;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.zip.Adler32;
import java.util.zip.Checksum;

/**
 * Abstract class for enabling dataset downloading and local caching.
 *
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public abstract class CacheableExtractableDataSetFetcher implements CacheableDataSet {

    @Override public String dataSetName(DataSetType set) { return ""; }
    @Override public String remoteDataUrl() { return remoteDataUrl(DataSetType.TRAIN); }
    @Override public long expectedChecksum() { return expectedChecksum(DataSetType.TRAIN); }
    public void downloadAndExtract() throws IOException { downloadAndExtract(DataSetType.TRAIN); }

    /**
     * Downloads and extracts the local dataset.
     *
     * @throws IOException
     */
    public void downloadAndExtract(DataSetType set) throws IOException {
        String localFilename = new File(remoteDataUrl(set)).getName();
        File tmpFile = new File(System.getProperty("java.io.tmpdir"), localFilename);
        File localCacheDir = getLocalCacheDir();

        // check empty cache
        if(localCacheDir.exists()) {
            File[] list = localCacheDir.listFiles();
            if(list == null || list.length == 0)
                localCacheDir.delete();
        }

        File localDestinationDir = new File(localCacheDir, dataSetName(set));
        if(!localDestinationDir.exists()) {
            localCacheDir.mkdirs();
            tmpFile.delete();
            log.info("Downloading dataset to " + tmpFile.getAbsolutePath());
            FileUtils.copyURLToFile(new URL(remoteDataUrl(set)), tmpFile);
        } else {
            //Directory exists and is non-empty - assume OK
            log.info("Using cached dataset at " + localCacheDir.getAbsolutePath());
            return;
        }

        if(expectedChecksum(set) != 0L) {
            log.info("Verifying download...");
            Checksum adler = new Adler32();
            FileUtils.checksum(tmpFile, adler);
            long localChecksum = adler.getValue();
            log.info("Checksum local is " + localChecksum + ", expecting "+expectedChecksum(set));

            if(expectedChecksum(set) != localChecksum) {
                log.error("Checksums do not match. Cleaning up files and failing...");
                tmpFile.delete();
                throw new IllegalStateException( "Dataset file failed checksum: " + tmpFile + " - expected checksum " + expectedChecksum(set)
                + " vs. actual checksum " + localChecksum + ". If this error persists, please open an issue at https://github.com/deeplearning4j/deeplearning4j.");
            }
        }

        try {
            ArchiveUtils.unzipFileTo(tmpFile.getAbsolutePath(), localCacheDir.getAbsolutePath());
        } catch (Throwable t){
            //Catch any errors during extraction, and delete the directory to avoid leaving the dir in an invalid state
            if(localCacheDir.exists())
                FileUtils.deleteDirectory(localCacheDir);
            throw t;
        }
    }

    protected File getLocalCacheDir(){
        return DL4JResources.getDirectory(ResourceType.DATASET, localCacheName());
    }

    /**
     * Returns a boolean indicating if the dataset is already cached locally.
     *
     * @return boolean
     */
    @Override
    public boolean isCached() {
        return getLocalCacheDir().exists();
    }


    protected static void deleteIfEmpty(File localCache){
        if(localCache.exists()) {
            File[] files = localCache.listFiles();
            if(files == null || files.length < 1){
                try {
                    FileUtils.deleteDirectory(localCache);
                } catch (IOException e){
                    //Ignore
                    log.debug("Error deleting directory: {}", localCache);
                }
            }
        }
    }
}
