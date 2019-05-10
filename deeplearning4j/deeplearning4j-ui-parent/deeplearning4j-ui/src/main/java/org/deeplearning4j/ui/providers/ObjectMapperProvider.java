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

package org.deeplearning4j.ui.providers;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.module.SimpleModule;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.serde.jackson.ndarray.NDArrayDeSerializer;
import org.nd4j.shade.serde.jackson.ndarray.NDArraySerializer;

import javax.ws.rs.ext.ContextResolver;
import javax.ws.rs.ext.Provider;

/**
 * @author Adam Gibson
 */
@Provider
public class ObjectMapperProvider implements ContextResolver<ObjectMapper> {
    @Override
    public ObjectMapper getContext(Class<?> type) {
        final ObjectMapper result = new ObjectMapper();
        result.registerModule(module());
        return result;
    }

    public static SimpleModule module() {
        SimpleModule module = new SimpleModule("nd4j");
        module.addDeserializer(INDArray.class, new NDArrayDeSerializer());
        module.addSerializer(INDArray.class, new NDArraySerializer());
        return module;
    }
}
