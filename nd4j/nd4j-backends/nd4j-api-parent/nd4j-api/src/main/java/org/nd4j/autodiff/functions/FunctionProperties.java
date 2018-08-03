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

package org.nd4j.autodiff.functions;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import com.google.flatbuffers.FlatBufferBuilder;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.graph.FlatNode;
import org.nd4j.graph.FlatProperties;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

@Data
@Slf4j
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class FunctionProperties {
    private String name;
    @Builder.Default private Map<String,Object> fieldNames = new LinkedHashMap<>();
    @Builder.Default private List<Integer> i = new ArrayList<>();
    @Builder.Default private List<Long> l = new ArrayList<>();
    @Builder.Default private List<Double> d = new ArrayList<>();
    @Builder.Default private List<INDArray> a = new ArrayList<>();

    /**
     * This method converts this FunctionProperties instance to FlatBuffers representation
     * @param bufferBuilder
     * @return
     */
    public int asFlatProperties(FlatBufferBuilder bufferBuilder) {
        int iname = bufferBuilder.createString(name);
        int ii = FlatProperties.createIVector(bufferBuilder, Ints.toArray(i));
        int il = FlatProperties.createLVector(bufferBuilder, Longs.toArray(l));
        int id = FlatProperties.createDVector(bufferBuilder, Doubles.toArray(d));

        int arrays[] = new int[a.size()];
        int cnt = 0;
        for (val array: a) {
            int off = array.toFlatArray(bufferBuilder);
            arrays[cnt++] = off;
        }

        int ia = FlatProperties.createAVector(bufferBuilder, arrays);

        return FlatProperties.createFlatProperties(bufferBuilder, iname, ii, il, id, ia);
    }

    /**
     * This method creates new FunctionProperties instance from FlatBuffers representation
     * @param properties
     * @return
     */
    public static FunctionProperties fromFlatProperties(FlatProperties properties) {
        val props = new FunctionProperties();

        for (int e = 0; e < properties.iLength(); e++)
            props.getI().add(properties.i(e));

        for (int e = 0; e < properties.lLength(); e++)
            props.getL().add(properties.l(e));

        for (int e = 0; e < properties.dLength(); e++)
            props.getD().add(properties.d(e));

        for (int e = 0; e < properties.iLength(); e++)
            props.getA().add(Nd4j.createFromFlatArray(properties.a(e)));

        return props;
    }

    /**
     * This method converts multiple FunctionProperties to FlatBuffers representation
     *
     * @param bufferBuilder
     * @param properties
     * @return
     */
    public static int asFlatProperties(FlatBufferBuilder bufferBuilder, Collection<FunctionProperties> properties) {
        int props[] = new int[properties.size()];

        int cnt = 0;
        for (val p: properties)
            props[cnt++] = p.asFlatProperties(bufferBuilder);

        return FlatNode.createPropertiesVector(bufferBuilder, props);
    }
}
