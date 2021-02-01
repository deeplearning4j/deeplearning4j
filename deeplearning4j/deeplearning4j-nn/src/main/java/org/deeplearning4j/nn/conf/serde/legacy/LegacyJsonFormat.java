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

package org.deeplearning4j.nn.conf.serde.legacy;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.graph.*;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.graph.rnn.ReverseTimeSeriesVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping1D;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.conf.layers.misc.ElementWiseMultiplicationLayer;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.conf.layers.util.MaskLayer;
import org.deeplearning4j.nn.conf.layers.util.MaskZeroLayer;
import org.deeplearning4j.nn.conf.layers.variational.*;
import org.deeplearning4j.nn.conf.preprocessor.*;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.*;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.databind.ObjectMapper;

public class LegacyJsonFormat {

    private LegacyJsonFormat(){ }

    /**
     * Get a mapper (minus general config) suitable for loading old format JSON - 1.0.0-alpha and before
     * @return Object mapper
     */
    public static ObjectMapper getMapper100alpha(){
        //After 1.0.0-alpha, we switched from wrapper object to @class for subtype information
        ObjectMapper om = new ObjectMapper();

        om.addMixIn(InputPreProcessor.class, InputPreProcessorMixin.class);
        om.addMixIn(GraphVertex.class, GraphVertexMixin.class);
        om.addMixIn(Layer.class, LayerMixin.class);
        om.addMixIn(ReconstructionDistribution.class, ReconstructionDistributionMixin.class);
        om.addMixIn(IActivation.class, IActivationMixin.class);
        om.addMixIn(ILossFunction.class, ILossFunctionMixin.class);

        return om;
    }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes(value = {@JsonSubTypes.Type(value = CnnToFeedForwardPreProcessor.class, name = "cnnToFeedForward"),
            @JsonSubTypes.Type(value = CnnToRnnPreProcessor.class, name = "cnnToRnn"),
            @JsonSubTypes.Type(value = ComposableInputPreProcessor.class, name = "composableInput"),
            @JsonSubTypes.Type(value = FeedForwardToCnnPreProcessor.class, name = "feedForwardToCnn"),
            @JsonSubTypes.Type(value = FeedForwardToRnnPreProcessor.class, name = "feedForwardToRnn"),
            @JsonSubTypes.Type(value = RnnToFeedForwardPreProcessor.class, name = "rnnToFeedForward"),
            @JsonSubTypes.Type(value = RnnToCnnPreProcessor.class, name = "rnnToCnn")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class InputPreProcessorMixin { }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes(value = {@JsonSubTypes.Type(value = ElementWiseVertex.class, name = "ElementWiseVertex"),
            @JsonSubTypes.Type(value = MergeVertex.class, name = "MergeVertex"),
            @JsonSubTypes.Type(value = SubsetVertex.class, name = "SubsetVertex"),
            @JsonSubTypes.Type(value = LayerVertex.class, name = "LayerVertex"),
            @JsonSubTypes.Type(value = LastTimeStepVertex.class, name = "LastTimeStepVertex"),
            @JsonSubTypes.Type(value = ReverseTimeSeriesVertex.class, name = "ReverseTimeSeriesVertex"),
            @JsonSubTypes.Type(value = DuplicateToTimeSeriesVertex.class, name = "DuplicateToTimeSeriesVertex"),
            @JsonSubTypes.Type(value = PreprocessorVertex.class, name = "PreprocessorVertex"),
            @JsonSubTypes.Type(value = StackVertex.class, name = "StackVertex"),
            @JsonSubTypes.Type(value = UnstackVertex.class, name = "UnstackVertex"),
            @JsonSubTypes.Type(value = L2Vertex.class, name = "L2Vertex"),
            @JsonSubTypes.Type(value = ScaleVertex.class, name = "ScaleVertex"),
            @JsonSubTypes.Type(value = L2NormalizeVertex.class, name = "L2NormalizeVertex")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class GraphVertexMixin{ }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes(value = {@JsonSubTypes.Type(value = AutoEncoder.class, name = "autoEncoder"),
            @JsonSubTypes.Type(value = ConvolutionLayer.class, name = "convolution"),
            @JsonSubTypes.Type(value = Convolution1DLayer.class, name = "convolution1d"),
            @JsonSubTypes.Type(value = GravesLSTM.class, name = "gravesLSTM"),
            @JsonSubTypes.Type(value = LSTM.class, name = "LSTM"),
            @JsonSubTypes.Type(value = GravesBidirectionalLSTM.class, name = "gravesBidirectionalLSTM"),
            @JsonSubTypes.Type(value = OutputLayer.class, name = "output"),
            @JsonSubTypes.Type(value = CenterLossOutputLayer.class, name = "CenterLossOutputLayer"),
            @JsonSubTypes.Type(value = RnnOutputLayer.class, name = "rnnoutput"),
            @JsonSubTypes.Type(value = LossLayer.class, name = "loss"),
            @JsonSubTypes.Type(value = DenseLayer.class, name = "dense"),
            @JsonSubTypes.Type(value = SubsamplingLayer.class, name = "subsampling"),
            @JsonSubTypes.Type(value = Subsampling1DLayer.class, name = "subsampling1d"),
            @JsonSubTypes.Type(value = BatchNormalization.class, name = "batchNormalization"),
            @JsonSubTypes.Type(value = LocalResponseNormalization.class, name = "localResponseNormalization"),
            @JsonSubTypes.Type(value = EmbeddingLayer.class, name = "embedding"),
            @JsonSubTypes.Type(value = ActivationLayer.class, name = "activation"),
            @JsonSubTypes.Type(value = VariationalAutoencoder.class, name = "VariationalAutoencoder"),
            @JsonSubTypes.Type(value = DropoutLayer.class, name = "dropout"),
            @JsonSubTypes.Type(value = GlobalPoolingLayer.class, name = "GlobalPooling"),
            @JsonSubTypes.Type(value = ZeroPaddingLayer.class, name = "zeroPadding"),
            @JsonSubTypes.Type(value = ZeroPadding1DLayer.class, name = "zeroPadding1d"),
            @JsonSubTypes.Type(value = FrozenLayer.class, name = "FrozenLayer"),
            @JsonSubTypes.Type(value = Upsampling2D.class, name = "Upsampling2D"),
            @JsonSubTypes.Type(value = Yolo2OutputLayer.class, name = "Yolo2OutputLayer"),
            @JsonSubTypes.Type(value = RnnLossLayer.class, name = "RnnLossLayer"),
            @JsonSubTypes.Type(value = CnnLossLayer.class, name = "CnnLossLayer"),
            @JsonSubTypes.Type(value = Bidirectional.class, name = "Bidirectional"),
            @JsonSubTypes.Type(value = SimpleRnn.class, name = "SimpleRnn"),
            @JsonSubTypes.Type(value = ElementWiseMultiplicationLayer.class, name = "ElementWiseMult"),
            @JsonSubTypes.Type(value = MaskLayer.class, name = "MaskLayer"),
            @JsonSubTypes.Type(value = MaskZeroLayer.class, name = "MaskZeroLayer"),
            @JsonSubTypes.Type(value = Cropping1D.class, name = "Cropping1D"),
            @JsonSubTypes.Type(value = Cropping2D.class, name = "Cropping2D")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class LayerMixin {}

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes(value = {@JsonSubTypes.Type(value = GaussianReconstructionDistribution.class, name = "Gaussian"),
            @JsonSubTypes.Type(value = BernoulliReconstructionDistribution.class, name = "Bernoulli"),
            @JsonSubTypes.Type(value = ExponentialReconstructionDistribution.class, name = "Exponential"),
            @JsonSubTypes.Type(value = CompositeReconstructionDistribution.class, name = "Composite"),
            @JsonSubTypes.Type(value = LossFunctionWrapper.class, name = "LossWrapper")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class ReconstructionDistributionMixin {}


    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes(value = {@JsonSubTypes.Type(value = ActivationCube.class, name = "Cube"),
            @JsonSubTypes.Type(value = ActivationELU.class, name = "ELU"),
            @JsonSubTypes.Type(value = ActivationHardSigmoid.class, name = "HardSigmoid"),
            @JsonSubTypes.Type(value = ActivationHardTanH.class, name = "HardTanh"),
            @JsonSubTypes.Type(value = ActivationIdentity.class, name = "Identity"),
            @JsonSubTypes.Type(value = ActivationLReLU.class, name = "LReLU"),
            @JsonSubTypes.Type(value = ActivationRationalTanh.class, name = "RationalTanh"),
            @JsonSubTypes.Type(value = ActivationRectifiedTanh.class, name = "RectifiedTanh"),
            @JsonSubTypes.Type(value = ActivationSELU.class, name = "SELU"),
            @JsonSubTypes.Type(value = ActivationSwish.class, name = "SWISH"),
            @JsonSubTypes.Type(value = ActivationReLU.class, name = "ReLU"),
            @JsonSubTypes.Type(value = ActivationRReLU.class, name = "RReLU"),
            @JsonSubTypes.Type(value = ActivationSigmoid.class, name = "Sigmoid"),
            @JsonSubTypes.Type(value = ActivationSoftmax.class, name = "Softmax"),
            @JsonSubTypes.Type(value = ActivationSoftPlus.class, name = "SoftPlus"),
            @JsonSubTypes.Type(value = ActivationSoftSign.class, name = "SoftSign"),
            @JsonSubTypes.Type(value = ActivationTanH.class, name = "TanH")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class IActivationMixin {}

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes(value = {@JsonSubTypes.Type(value = LossBinaryXENT.class, name = "BinaryXENT"),
            @JsonSubTypes.Type(value = LossCosineProximity.class, name = "CosineProximity"),
            @JsonSubTypes.Type(value = LossHinge.class, name = "Hinge"),
            @JsonSubTypes.Type(value = LossKLD.class, name = "KLD"),
            @JsonSubTypes.Type(value = LossMAE.class, name = "MAE"),
            @JsonSubTypes.Type(value = LossL1.class, name = "L1"),
            @JsonSubTypes.Type(value = LossMAPE.class, name = "MAPE"),
            @JsonSubTypes.Type(value = LossMCXENT.class, name = "MCXENT"),
            @JsonSubTypes.Type(value = LossMSE.class, name = "MSE"),
            @JsonSubTypes.Type(value = LossL2.class, name = "L2"),
            @JsonSubTypes.Type(value = LossMSLE.class, name = "MSLE"),
            @JsonSubTypes.Type(value = LossNegativeLogLikelihood.class, name = "NegativeLogLikelihood"),
            @JsonSubTypes.Type(value = LossPoisson.class, name = "Poisson"),
            @JsonSubTypes.Type(value = LossSquaredHinge.class, name = "SquaredHinge"),
            @JsonSubTypes.Type(value = LossFMeasure.class, name = "FMeasure")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class ILossFunctionMixin {}
}
