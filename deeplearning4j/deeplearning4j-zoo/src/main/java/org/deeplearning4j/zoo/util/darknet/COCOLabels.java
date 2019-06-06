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

package org.deeplearning4j.zoo.util.darknet;

import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.zoo.util.BaseLabels;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;

/**
 * Helper class that returns label descriptions for YOLO models trained with <a href="http://cocodataset.org/">COCO</a>.
 *
 * @author saudet
 */
public class COCOLabels extends BaseLabels {

    public COCOLabels() throws IOException {
        super("coco.names");
    }

    @Override
    protected URL getURL() {
        try {
            return DL4JResources.getURL("resources/darknet/coco.names");
        } catch (MalformedURLException e){
            throw new RuntimeException(e);
        }
    }

    @Override
    protected String resourceName() {
        return "darknet";
    }

    @Override
    protected String resourceMD5() {
        return "4caf6834300c8b2ff19964b36e54d637";
    }
}
