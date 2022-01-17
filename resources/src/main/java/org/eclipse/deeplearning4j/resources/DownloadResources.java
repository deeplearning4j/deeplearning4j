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

/**
 * Top level namespace for handling downloaded resources.
 * Contains utility classes for working with different resources.
 */
public class DownloadResources {

    /**
     * Create a {@link CustomResource}
     * @param fileName the name of the file to download
     * @param localCacheDirectory the cache directory to store it
     * @param rootUrl the root url of the download
     * @return the created {@link CustomResource}
     */
    public static CustomResource createCustomResource(String fileName,String localCacheDirectory,String rootUrl) {
        return new CustomResource(fileName,localCacheDirectory,rootUrl);
    }

    /**
     * Create a legacy dl4j zoo model
     * @param fileName the file name of the model to download
     * @param modelName the name of the model to download
     * @return
     */
    public static Dl4jZooResource createLegacyZooResource(String fileName,String modelName) {
        return new Dl4jZooResource(fileName,modelName);
    }

    /**
     * Create an {@link DataSetResource}
     * @param fileName the name of the file to download
     * @return
     */
    public static DataSetResource createDatasetResource(String fileName,String localCacheDirectory,String rootUrl) {
        return new DataSetResource(fileName,localCacheDirectory,rootUrl);
    }

    /**
     * Create a {@link OmniHubResource} from the given file
     * @param fileName the name of the file to download
     * @return the created resource
     */
    public static OmniHubResource createOmnihubResource(String fileName) {
        return new OmniHubResource(fileName);
    }


}
