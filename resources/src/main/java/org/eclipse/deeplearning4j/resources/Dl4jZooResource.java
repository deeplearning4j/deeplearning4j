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

import org.deeplearning4j.common.resources.DL4JResources;
import org.eclipse.deeplearning4j.omnihub.OmniHubUtils;

import java.io.File;

public class Dl4jZooResource extends BaseResource {

    private String modelName;


    public Dl4jZooResource(String fileName,String archiveName,String modelName) {
        super(fileName,archiveName);
        this.modelName = modelName;
    }

    public Dl4jZooResource(String fileName,String modelName) {
        this(fileName,"",modelName);
    }


    @Override
    public String fileName() {
        return fileName;
    }

    @Override
    public String rootUrl() {
        return DL4JResources.getBaseDownloadURL();
    }

    @Override
    public File localCacheDirectory() {
        File rootCacheDir = DL4JResources.getDirectory(org.deeplearning4j.common.resources.ResourceType.ZOO_MODEL, modelName);
        return rootCacheDir;
    }

    @Override
    public ResourceType resourceType() {
        return ResourceType.LEGACY_ZOO;
    }


}
