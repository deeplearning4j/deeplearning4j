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

import java.io.File;

public class CustomResource extends BaseResource {

    private String localCacheDirectory;
    private String rootUrl;

    public CustomResource(String fileName, String localCacheDirectory, String rootUrl) {
        super(fileName);
        this.localCacheDirectory = localCacheDirectory;
        this.rootUrl = rootUrl;
    }

    @Override
    public String fileName() {
        return fileName;
    }

    @Override
    public String rootUrl() {
        return rootUrl;
    }

    @Override
    public File localCacheDirectory() {
        return new File(localCacheDirectory);
    }

    @Override
    public ResourceType resourceType() {
        return ResourceType.CUSTOM;
    }
}
