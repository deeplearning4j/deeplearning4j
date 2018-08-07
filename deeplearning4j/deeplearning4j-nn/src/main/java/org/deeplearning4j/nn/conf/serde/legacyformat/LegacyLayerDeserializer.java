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
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping1D;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.conf.layers.misc.ElementWiseMultiplicationLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.conf.layers.util.MaskLayer;
import org.deeplearning4j.nn.conf.layers.util.MaskZeroLayer;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.conf.serde.JsonMappers;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.nd4j.serde.json.BaseLegacyDeserializer;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.*;

/**
 * Deserializer for Layer JSON in legacy format - see {@link BaseLegacyDeserializer}
 *
 * @author Alex Black
 */
public class LegacyLayerDeserializer extends BaseLegacyDeserializer<Layer> {

    private static final Map<String,String> LEGACY_NAMES = new HashMap<>();

    static {
        LEGACY_NAMES.put("autoEncoder", AutoEncoder.class.getName());
        LEGACY_NAMES.put("convolution", ConvolutionLayer.class.getName());
        LEGACY_NAMES.put("convolution1d", Convolution1DLayer.class.getName());
        LEGACY_NAMES.put("gravesLSTM", GravesLSTM.class.getName());
        LEGACY_NAMES.put("LSTM", LSTM.class.getName());
        LEGACY_NAMES.put("gravesBidirectionalLSTM", GravesBidirectionalLSTM.class.getName());
        LEGACY_NAMES.put("output", OutputLayer.class.getName());
        LEGACY_NAMES.put("CenterLossOutputLayer", CenterLossOutputLayer.class.getName());
        LEGACY_NAMES.put("rnnoutput", RnnOutputLayer.class.getName());
        LEGACY_NAMES.put("loss", LossLayer.class.getName());
        LEGACY_NAMES.put("dense", DenseLayer.class.getName());
        LEGACY_NAMES.put("subsampling", SubsamplingLayer.class.getName());
        LEGACY_NAMES.put("subsampling1d", Subsampling1DLayer.class.getName());
        LEGACY_NAMES.put("batchNormalization", BatchNormalization.class.getName());
        LEGACY_NAMES.put("localResponseNormalization", LocalResponseNormalization.class.getName());
        LEGACY_NAMES.put("embedding", EmbeddingLayer.class.getName());
        LEGACY_NAMES.put("activation", ActivationLayer.class.getName());
        LEGACY_NAMES.put("VariationalAutoencoder", VariationalAutoencoder.class.getName());
        LEGACY_NAMES.put("dropout", DropoutLayer.class.getName());
        LEGACY_NAMES.put("GlobalPooling", GlobalPoolingLayer.class.getName());
        LEGACY_NAMES.put("zeroPadding", ZeroPaddingLayer.class.getName());
        LEGACY_NAMES.put("zeroPadding1d", ZeroPadding1DLayer.class.getName());
        LEGACY_NAMES.put("FrozenLayer", FrozenLayer.class.getName());
        LEGACY_NAMES.put("Upsampling2D", Upsampling2D.class.getName());
        LEGACY_NAMES.put("Yolo2OutputLayer", Yolo2OutputLayer.class.getName());
        LEGACY_NAMES.put("RnnLossLayer", RnnLossLayer.class.getName());
        LEGACY_NAMES.put("CnnLossLayer", CnnLossLayer.class.getName());
        LEGACY_NAMES.put("Bidirectional", Bidirectional.class.getName());
        LEGACY_NAMES.put("SimpleRnn", SimpleRnn.class.getName());
        LEGACY_NAMES.put("ElementWiseMult", ElementWiseMultiplicationLayer.class.getName());
        LEGACY_NAMES.put("MaskLayer", MaskLayer.class.getName());
        LEGACY_NAMES.put("MaskZeroLayer", MaskZeroLayer.class.getName());
        LEGACY_NAMES.put("Cropping1D", Cropping1D.class.getName());
        LEGACY_NAMES.put("Cropping2D", Cropping2D.class.getName());

        //The following didn't previously have subtype annotations - hence will be using default name (class simple name)
        LEGACY_NAMES.put("LastTimeStep", LastTimeStep.class.getName());
        LEGACY_NAMES.put("SpaceToDepthLayer", SpaceToDepthLayer.class.getName());
        LEGACY_NAMES.put("SpaceToBatchLayer", SpaceToBatchLayer.class.getName());
    }


    @Override
    public Map<String, String> getLegacyNamesMap() {
        return LEGACY_NAMES;
    }

    @Override
    public ObjectMapper getLegacyJsonMapper() {
        return JsonMappers.getJsonMapperLegacyFormatLayer();
    }

    @Override
    public Class<?> getDeserializedType() {
        return Layer.class;
    }

    public static void registerLegacyClassDefaultName(@NonNull Class<? extends Layer> clazz){
        registerLegacyClassSpecifiedName(clazz.getSimpleName(), clazz);
    }

    public static void registerLegacyClassSpecifiedName(@NonNull String name, @NonNull Class<? extends Layer> clazz){
        LEGACY_NAMES.put(name, clazz.getName());
    }
}
