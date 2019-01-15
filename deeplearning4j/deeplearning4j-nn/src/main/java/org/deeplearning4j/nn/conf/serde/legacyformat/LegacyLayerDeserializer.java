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
import org.deeplearning4j.nn.conf.layers.convolutional.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.convolutional.Convolution1DLayer;
import org.deeplearning4j.nn.conf.layers.convolutional.Convolution2DLayer;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping1DLayer;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2DLayer;
import org.deeplearning4j.nn.conf.layers.convolutional.SpaceToBatchLayer;
import org.deeplearning4j.nn.conf.layers.convolutional.SpaceToDepthLayer;
import org.deeplearning4j.nn.conf.layers.convolutional.ZeroPadding1DLayer;
import org.deeplearning4j.nn.conf.layers.convolutional.ZeroPadding2DLayer;
import org.deeplearning4j.nn.conf.layers.convolutional.subsampling.Subsampling1DLayer;
import org.deeplearning4j.nn.conf.layers.convolutional.subsampling.Subsampling2DLayer;
import org.deeplearning4j.nn.conf.layers.convolutional.upsampling.Upsampling2DLayer;
import org.deeplearning4j.nn.conf.layers.feedforeward.autoencoder.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.feedforeward.dense.DenseLayer;
import org.deeplearning4j.nn.conf.layers.feedforeward.elementwise.ElementWiseMultiplicationLayer;
import org.deeplearning4j.nn.conf.layers.feedforeward.embedding.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.deeplearning4j.nn.conf.layers.normalization.BatchNormalizationLayer;
import org.deeplearning4j.nn.conf.layers.normalization.LocalResponseNormalizationLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.conf.layers.pooling.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.BidirectionalLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.GravesBidirectionalLSTMLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.GravesLSTMLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.LSTMLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStepLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.RnnLossLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnnLayer;
import org.deeplearning4j.nn.conf.layers.training.CenterLossOutputLayer;
import org.deeplearning4j.nn.conf.layers.util.MaskLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.MaskZeroLayer;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.conf.serde.JsonMappers;
import org.nd4j.serde.json.BaseLegacyDeserializer;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.util.HashMap;
import java.util.Map;

/**
 * Deserializer for Layer JSON in legacy format - see {@link BaseLegacyDeserializer}
 *
 * @author Alex Black
 */
public class LegacyLayerDeserializer extends BaseLegacyDeserializer<Layer> {

    private static final Map<String,String> LEGACY_NAMES = new HashMap<>();

    static {
        LEGACY_NAMES.put("autoEncoder", AutoEncoder.class.getName());
        LEGACY_NAMES.put("convolution", Convolution2DLayer.class.getName());
        LEGACY_NAMES.put("convolution1d", Convolution1DLayer.class.getName());
        LEGACY_NAMES.put("gravesLSTM", GravesLSTMLayer.class.getName());
        LEGACY_NAMES.put("LSTMLayer", LSTMLayer.class.getName());
        LEGACY_NAMES.put("gravesBidirectionalLSTM", GravesBidirectionalLSTMLayer.class.getName());
        LEGACY_NAMES.put("output", OutputLayer.class.getName());
        LEGACY_NAMES.put("CenterLossOutputLayer", CenterLossOutputLayer.class.getName());
        LEGACY_NAMES.put("rnnoutput", RnnOutputLayer.class.getName());
        LEGACY_NAMES.put("loss", LossLayer.class.getName());
        LEGACY_NAMES.put("dense", DenseLayer.class.getName());
        LEGACY_NAMES.put("subsampling", Subsampling2DLayer.class.getName());
        LEGACY_NAMES.put("subsampling1d", Subsampling1DLayer.class.getName());
        LEGACY_NAMES.put("batchNormalization", BatchNormalizationLayer.class.getName());
        LEGACY_NAMES.put("localResponseNormalization", LocalResponseNormalizationLayer.class.getName());
        LEGACY_NAMES.put("embedding", EmbeddingLayer.class.getName());
        LEGACY_NAMES.put("activation", ActivationLayer.class.getName());
        LEGACY_NAMES.put("VariationalAutoencoder", VariationalAutoencoder.class.getName());
        LEGACY_NAMES.put("dropout", DropoutLayer.class.getName());
        LEGACY_NAMES.put("GlobalPooling", GlobalPoolingLayer.class.getName());
        LEGACY_NAMES.put("zeroPadding", ZeroPadding2DLayer.class.getName());
        LEGACY_NAMES.put("zeroPadding1d", ZeroPadding1DLayer.class.getName());
        LEGACY_NAMES.put("FrozenLayer", FrozenLayer.class.getName());
        LEGACY_NAMES.put("Upsampling2DLayer", Upsampling2DLayer.class.getName());
        LEGACY_NAMES.put("Yolo2OutputLayer", Yolo2OutputLayer.class.getName());
        LEGACY_NAMES.put("RnnLossLayer", RnnLossLayer.class.getName());
        LEGACY_NAMES.put("CnnLossLayer", CnnLossLayer.class.getName());
        LEGACY_NAMES.put("BidirectionalLayer", BidirectionalLayer.class.getName());
        LEGACY_NAMES.put("SimpleRnnLayer", SimpleRnnLayer.class.getName());
        LEGACY_NAMES.put("ElementWiseMult", ElementWiseMultiplicationLayer.class.getName());
        LEGACY_NAMES.put("MaskLayer", MaskLayer.class.getName());
        LEGACY_NAMES.put("MaskZeroLayer", MaskZeroLayer.class.getName());
        LEGACY_NAMES.put("Cropping1DLayer", Cropping1DLayer.class.getName());
        LEGACY_NAMES.put("Cropping2DLayer", Cropping2DLayer.class.getName());

        //The following didn't previously have subtype annotations - hence will be using default name (class simple name)
        LEGACY_NAMES.put("LastTimeStepLayer", LastTimeStepLayer.class.getName());
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
