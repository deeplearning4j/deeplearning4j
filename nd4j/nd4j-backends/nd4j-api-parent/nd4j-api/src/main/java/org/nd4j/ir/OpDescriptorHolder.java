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
package org.nd4j.ir;

import lombok.val;
import org.apache.commons.io.IOUtils;
import org.nd4j.common.config.ND4JClassLoading;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.shade.protobuf.TextFormat;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.*;

/**
 * A utility class for accessing the nd4j op descriptors.
 * May override default definition of {@link #nd4jFileNameTextDefault}
 * with the system property {@link #nd4jFileSpecifierProperty}
 * @author Adam Gibson
 */
public class OpDescriptorHolder {

    public static String  nd4jFileNameTextDefault = "/nd4j-op-def.pbtxt";
    public static String nd4jFileSpecifierProperty = "samediff.import.nd4jdescriptors";
    public static OpNamespace.OpDescriptorList INSTANCE;
    private static Map<String, OpNamespace.OpDescriptor> opDescriptorByName;

    static {
        try {
            INSTANCE = nd4jOpList();
        } catch (IOException e) {
            e.printStackTrace();
        }

        opDescriptorByName = new LinkedHashMap<>();
        for(int i = 0; i < INSTANCE.getOpListCount(); i++) {
            opDescriptorByName.put(INSTANCE.getOpList(i).getName(),INSTANCE.getOpList(i));
        }

    }

    /**
     * Return the {@link OpNamespace.OpDescriptor}
     * for a given op name
     * @param name the name of the op to get the descriptor for
     * @return the desired op descriptor or null if it does not exist
     */
    public static OpNamespace.OpDescriptor descriptorForOpName(String name) {
        return opDescriptorByName.get(name);
    }

    /**
     * Returns an singleton of the {@link #nd4jOpList()}
     * result, useful for preventing repeated I/O.
     * @return
     */
    public static OpNamespace.OpDescriptorList opList() {
        return INSTANCE;
    }

    /**
     * Get the nd4j op list {@link OpNamespace.OpDescriptorList} for serialization.
     * Useful for saving and loading {@link org.nd4j.linalg.api.ops.DynamicCustomOp}
     * @return the static list of descriptors from the nd4j classpath.
     * @throws IOException
     */
    public static OpNamespace.OpDescriptorList nd4jOpList() throws IOException  {
        val fileName = System.getProperty(nd4jFileSpecifierProperty, nd4jFileNameTextDefault);
        val nd4jOpDescriptorResourceStream = new ClassPathResource(fileName, ND4JClassLoading.getNd4jClassloader()).getInputStream();
        val resourceString = IOUtils.toString(nd4jOpDescriptorResourceStream, Charset.defaultCharset());
        val descriptorListBuilder = OpNamespace.OpDescriptorList.newBuilder();
        TextFormat.merge(resourceString,descriptorListBuilder);
        val ret = descriptorListBuilder.build();
        val mutableList = new ArrayList<>(ret.getOpListList());
        Collections.sort(mutableList, Comparator.comparing(OpNamespace.OpDescriptor::getName));

        val newResultBuilder = OpNamespace.OpDescriptorList.newBuilder();
        newResultBuilder.addAllOpList(mutableList);
        return newResultBuilder.build();
    }

}
