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
package org.eclipse.deeplearning4j.resources;

import lombok.SneakyThrows;
import org.apache.commons.io.FileUtils;
import org.nd4j.common.resources.Downloader;

import java.io.File;
import java.io.IOException;
import java.net.URI;

/**
 * A downloadable resource represents a resource
 * that is downloadable and cached in a specified
 * directory. This will usually be in the user's home directory
 * under a hidden dot directory.
 */
public interface DownloadableResource {


    /**
     * The name of the file to be downloaded
     * or if an archive, the naem of the extracted file
     * @return
     */
    String fileName();

    /**
     * The name of the archive file to be downloaded
     * @return
     */
    String archiveFileName();


    /**
     * The root url to the resource.
     * This is going to be the directory name of
     * where the file lives.
     * @return
     */
    String rootUrl();

    /**
     * The storage location representing the directory cache
     * @return
     */
    File localCacheDirectory();

    /**
     * The local path to the file. This will usually be
     * {@link #localCacheDirectory()} + {@link #fileName()}
     * @return
     */
    default File localPath() {
        return new File(localCacheDirectory(),fileName());
    }
    /**
     * Md5sum of the file.
     * Used for verification maybe empty
     * to avoid running verification.
     * @return
     */
    default String md5Sum() {
        return "";
    }




    @SneakyThrows
    default void download(boolean archive,int retries,int connectionTimeout,int readTimeout) {
        if(archive) {
            localCacheDirectory().mkdirs();
            Downloader.downloadAndExtract(archiveFileName(),
                    URI.create(rootUrl() + "/" + archiveFileName()).toURL(),
                    new File(localCacheDirectory(),archiveFileName()),
                    localPath(),
                    md5Sum(),
                    retries, connectionTimeout,
                    readTimeout);
        } else {
            Downloader.download(fileName(),
                    URI.create(rootUrl() + "/" + fileName()).toURL(),
                    localPath(),
                    md5Sum(),
                    retries,
                    connectionTimeout,
                    readTimeout);
        }
    }


    /**
     * Returns the resource type for this resource
     *
     * @return
     */
    ResourceType resourceType();


    /**
     *
     * @return
     */
    default boolean existsLocally() {
        return localPath().exists();
    }

    default void delete() throws IOException {
        if(localPath().isDirectory())
            FileUtils.deleteDirectory(localPath());
        else
            FileUtils.forceDelete(localPath());
    }

}
