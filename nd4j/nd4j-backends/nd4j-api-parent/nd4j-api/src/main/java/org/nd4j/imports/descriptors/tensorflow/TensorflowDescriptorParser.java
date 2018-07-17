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

package org.nd4j.imports.descriptors.tensorflow;

import com.github.os72.protobuf351.TextFormat;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.io.ClassPathResource;
import org.tensorflow.framework.OpDef;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TensorflowDescriptorParser {

    /**
     * Get the op descriptors for tensorflow
     * @return the op descriptors for tensorflow
     * @throws Exception
     */
    public static Map<String,OpDef> opDescs() throws Exception {
        InputStream contents = new ClassPathResource("ops.proto").getInputStream();
        try (BufferedInputStream bis2 = new BufferedInputStream(contents); BufferedReader reader = new BufferedReader(new InputStreamReader(bis2))) {
            org.tensorflow.framework.OpList.Builder builder = org.tensorflow.framework.OpList.newBuilder();

            StringBuilder str = new StringBuilder();
            String line = null;
            while ((line = reader.readLine()) != null) {
                str.append(line);//.append("\n");
            }


            TextFormat.getParser().merge(str.toString(), builder);
            List<OpDef> list =  builder.getOpList();
            Map<String,OpDef> map = new HashMap<>();
            for(OpDef opDef : list) {
                map.put(opDef.getName(),opDef);
            }

            return map;

        } catch (Exception e2) {
            e2.printStackTrace();
        }

        throw new ND4JIllegalStateException("Unable to load tensorflow descriptors!");

    }


}
