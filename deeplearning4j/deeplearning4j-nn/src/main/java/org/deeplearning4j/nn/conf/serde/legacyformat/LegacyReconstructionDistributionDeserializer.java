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

package org.deeplearning4j.nn.conf.serde.legacyformat;

import lombok.NonNull;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.layers.variational.*;
import org.deeplearning4j.nn.conf.preprocessor.*;
import org.deeplearning4j.nn.conf.serde.JsonMappers;
import org.nd4j.serde.json.BaseLegacyDeserializer;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.util.HashMap;
import java.util.Map;

/**
 * Deserializer for ReconstructionDistribution JSON in legacy format - see {@link BaseLegacyDeserializer}
 *
 * @author Alex Black
 */
public class LegacyReconstructionDistributionDeserializer extends BaseLegacyDeserializer<ReconstructionDistribution> {

    private static final Map<String,String> LEGACY_NAMES = new HashMap<>();

    static {
        LEGACY_NAMES.put("Gaussian", GaussianReconstructionDistribution.class.getName());
        LEGACY_NAMES.put("Bernoulli", BernoulliReconstructionDistribution.class.getName());
        LEGACY_NAMES.put("Exponential", ExponentialReconstructionDistribution.class.getName());
        LEGACY_NAMES.put("Composite", CompositeReconstructionDistribution.class.getName());
        LEGACY_NAMES.put("LossWrapper", LossFunctionWrapper.class.getName());
    }


    @Override
    public Map<String, String> getLegacyNamesMap() {
        return LEGACY_NAMES;
    }

    @Override
    public ObjectMapper getLegacyJsonMapper() {
        return JsonMappers.getJsonMapperLegacyFormatReconstruction();
    }

    @Override
    public Class<?> getDeserializedType() {
        return ReconstructionDistribution.class;
    }

    public static void registerLegacyClassDefaultName(@NonNull Class<? extends ReconstructionDistribution> clazz){
        registerLegacyClassSpecifiedName(clazz.getSimpleName(), clazz);
    }

    public static void registerLegacyClassSpecifiedName(@NonNull String name, @NonNull Class<? extends ReconstructionDistribution> clazz){
        LEGACY_NAMES.put(name, clazz.getName());
    }
}
