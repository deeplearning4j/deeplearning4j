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

package org.nd4j.imports.descriptors.onnx;

import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Load all of the onnx op descriptors from the classpath.
 *
 * @author Adam Gibson
 */
public class OnnxDescriptorParser {


    /**
     * Get the onnx op descriptors by name
     * @return the onnx op descriptors by name
     * @throws Exception
     */
    public static Map<String,OpDescriptor> onnxOpDescriptors() throws Exception {
        try(InputStream is = new ClassPathResource("onnxops.json").getInputStream()) {
            ObjectMapper objectMapper = new ObjectMapper();
            OnnxDescriptor opDescriptor = objectMapper.readValue(is,OnnxDescriptor.class);
            Map<String,OpDescriptor> descriptorMap = new HashMap<>();
            for(OpDescriptor descriptor : opDescriptor.getDescriptors()) {
                descriptorMap.put(descriptor.getName(),descriptor);
            }



            return descriptorMap;
        }
    }


}
