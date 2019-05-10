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

package org.nd4j.serde.json;

import lombok.NonNull;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.util.HashMap;
import java.util.Map;

/**
 * Deserializer for IActivation JSON in legacy format - see {@link BaseLegacyDeserializer}
 *
 * @author Alex Black
 */
public class LegacyIActivationDeserializer extends BaseLegacyDeserializer<IActivation> {
    private static final Map<String,String> LEGACY_NAMES = new HashMap<>();

    private static ObjectMapper legacyMapper;

    public static void setLegacyJsonMapper(ObjectMapper mapper){
        legacyMapper = mapper;
    }

    static {
        LEGACY_NAMES.put("Cube", ActivationCube.class.getName());
        LEGACY_NAMES.put("ELU", ActivationELU.class.getName());
        LEGACY_NAMES.put("HardSigmoid", ActivationHardSigmoid.class.getName());
        LEGACY_NAMES.put("HardTanh", ActivationHardTanH.class.getName());
        LEGACY_NAMES.put("Identity", ActivationIdentity.class.getName());
        LEGACY_NAMES.put("LReLU", ActivationLReLU.class.getName());
        LEGACY_NAMES.put("RationalTanh", ActivationRationalTanh.class.getName());
        LEGACY_NAMES.put("RectifiedTanh", ActivationRectifiedTanh.class.getName());
        LEGACY_NAMES.put("SELU", ActivationSELU.class.getName());
        LEGACY_NAMES.put("SWISH", ActivationSwish.class.getName());
        LEGACY_NAMES.put("ReLU", ActivationReLU.class.getName());
        LEGACY_NAMES.put("RReLU", ActivationRReLU.class.getName());
        LEGACY_NAMES.put("Sigmoid", ActivationSigmoid.class.getName());
        LEGACY_NAMES.put("Softmax", ActivationSoftmax.class.getName());
        LEGACY_NAMES.put("Softplus", ActivationSoftPlus.class.getName());
        LEGACY_NAMES.put("SoftSign", ActivationSoftSign.class.getName());
        LEGACY_NAMES.put("TanH", ActivationTanH.class.getName());

        //The following didn't previously have subtype annotations - hence will be using default name (class simple name)
        LEGACY_NAMES.put(ActivationReLU6.class.getSimpleName(), ActivationReLU6.class.getName());
    }


    @Override
    public Map<String, String> getLegacyNamesMap() {
        return LEGACY_NAMES;
    }

    @Override
    public ObjectMapper getLegacyJsonMapper() {
        return legacyMapper;
    }

    @Override
    public Class<?> getDeserializedType() {
        return IActivation.class;
    }

    public static void registerLegacyClassDefaultName(@NonNull Class<? extends IActivation> clazz){
        registerLegacyClassSpecifiedName(clazz.getSimpleName(), clazz);
    }

    public static void registerLegacyClassSpecifiedName(@NonNull String name, @NonNull Class<? extends IActivation> clazz){
        LEGACY_NAMES.put(name, clazz.getName());
    }
}
