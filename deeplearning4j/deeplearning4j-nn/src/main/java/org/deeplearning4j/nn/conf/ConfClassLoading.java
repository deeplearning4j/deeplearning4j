
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
package org.deeplearning4j.nn.conf;

import org.deeplearning4j.nn.conf.constraint.MaxNormConstraint;
import org.deeplearning4j.nn.conf.constraint.MinMaxNormConstraint;
import org.deeplearning4j.nn.conf.constraint.NonNegativeConstraint;
import org.deeplearning4j.nn.conf.constraint.UnitNormConstraint;
import org.deeplearning4j.nn.conf.distribution.*;
import org.deeplearning4j.nn.conf.dropout.AlphaDropout;
import org.deeplearning4j.nn.conf.dropout.GaussianDropout;
import org.deeplearning4j.nn.conf.dropout.GaussianNoise;
import org.deeplearning4j.nn.conf.dropout.SpatialDropout;
import org.deeplearning4j.nn.conf.graph.*;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.Convolution1DLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.ZeroPadding1DLayer;
import org.deeplearning4j.nn.conf.layers.ZeroPadding3DLayer;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping1D;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping3D;
import org.deeplearning4j.nn.conf.layers.misc.ElementWiseMultiplicationLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.conf.layers.recurrent.TimeDistributed;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaVertex;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffOutputLayer;
import org.deeplearning4j.nn.conf.layers.util.MaskZeroLayer;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.deeplearning4j.nn.layers.RepeatVector;
import org.deeplearning4j.nn.layers.convolution.*;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.layers.util.IdentityLayer;
import org.deeplearning4j.nn.layers.util.MaskLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.common.tools.ClassInitializerUtil;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.api.ops.impl.layers.convolution.DepthToSpace;
import org.nd4j.linalg.api.ops.impl.transforms.custom.BatchToSpace;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.*;

public class ConfClassLoading {
    private static AtomicBoolean invoked = new AtomicBoolean(false);

    public static void loadConfigClasses() throws ClassNotFoundException {
        if(invoked.get()) return;

        ClassInitializerUtil.tryLoadClasses(MultiLayerConfiguration.class,
                MultiLayerConfiguration.Builder.class,
                LossFunctions.class,
                ILossFunction.class,
                LossMSE.class,
                LossMAE.class,
                LossBinaryXENT.class,
                LossFMeasure.class,
                LossSparseMCXENT.class,
                LossNegativeLogLikelihood.class,
                LossMCXENT.class,
                LossKLD.class,
                LossL1.class,
                LossL2.class,
                LossHinge.class,
                LossSquaredHinge.class,
                LossCosineProximity.class,
                LossPoisson.class,
                LossMAPE.class,
                LossMSLE.class,
                LossL2.class,
                LossL1.class,
                LossWasserstein.class,
                MultiLayerNetwork.class,
                NeuralNetConfiguration.class,
                NeuralNetConfiguration.Builder.class,
                ComputationGraphConfiguration.class,
                ComputationGraphConfiguration.GraphBuilder.class,
                ComputationGraph.class,
                Layer.class,
                Layer.Builder.class,
                FeedForwardLayer.class,
                BaseOutputLayer.class,
                BaseLayer.class,
                ConvolutionLayer.class,
                ConvolutionLayer.Builder.class,
                Convolution1DLayer.class,
                Convolution1DLayer.Builder.class,
                Convolution3DLayer.class,
                Class.forName("org.deeplearning4j.nn.conf.layers.SubsamplingLayer$1"),
                org.nd4j.linalg.util.LongUtils.class,
                DifferentialFunction.class,
                ConvolutionMode.class,
                CNN2DFormat.class,
                PoolingType.class,
                SubsamplingLayer.class,
                SubsamplingLayer.Builder.class,
                PrimaryCapsules.class,
                CapsuleLayer.class,
                RecurrentAttentionLayer.class,
                //activations,
                ActivationCube.class,
                ActivationELU.class,
                ActivationHardSigmoid.class,
                ActivationHardTanH.class,
                ActivationIdentity.class,
                ActivationLReLU.class,
                ActivationRationalTanh.class,
                ActivationRectifiedTanh.class,
                ActivationReLU.class,
                ActivationReLU6.class,
                ActivationSELU.class,
                ActivationSwish.class,
                ActivationRReLU.class,
                ActivationSigmoid.class,
                ActivationSoftmax.class,
                ActivationSoftPlus.class,
                ActivationSoftSign.class,
                ActivationTanH.class,
                ActivationThresholdedReLU.class,
                ActivationGELU.class,
                ActivationMish.class,



                //normalizations
                MaxNormConstraint.class,
                MinMaxNormConstraint.class,
                NonNegativeConstraint.class,
                UnitNormConstraint.class,
                //distributions
                BinomialDistribution.class,
                ConstantDistribution.class,
                LogNormalDistribution.class,
                NormalDistribution.class,
                OrthogonalDistribution.class,
                TruncatedNormalDistribution.class,
                UniformDistribution.class,

                //vertices:
                AttentionVertex.class,
                DotProductAttentionLayer.class,
                ElementWiseVertex.class,
                GraphVertex.class,
                L2Vertex.class,
                MergeVertex.class,
                PreprocessorVertex.class,
                ReshapeVertex.class,
                ScaleVertex.class,
                ShiftVertex.class,
                SubsetVertex.class,
                UnstackVertex.class,
                StackVertex.class,
                LastTimeStepVertex.class,
                DuplicateToTimeSeriesVertex.class,
                PreprocessorVertex.class,

                //samediff
                SameDiffLambdaLayer.class,
                SameDiffLambdaVertex.class,
                SameDiffLayer.class,
                SameDiffOutputLayer.class,



                //dropout
                AlphaDropout.class,
                GaussianDropout.class,
                GaussianNoise.class,
                SpatialDropout.class,

                //layers
                DenseLayer.class,
                AutoEncoder.class,
                VariationalAutoencoder.class,
                ElementWiseMultiplicationLayer.class,
                PReLULayer.class,
                EmbeddingLayer.class,
                OutputLayer.class,
                EmbeddingSequenceLayer.class,
                BatchNormalization.class,
                LocalResponseNormalization.class,
                Yolo2OutputLayer.class,
                IdentityLayer.class,
                MaskLayer.class,
                OCNNOutputLayer.class,
                GlobalPoolingLayer.class,
                LastTimeStep.class,
                MaskZeroLayer.class,
                SimpleRnn.class,
                TimeDistributed.class,
                Bidirectional.class,
                ActivationLayer.class,
                DropoutLayer.class,
                FrozenLayer.class,
                RepeatVector.class,
                Subsampling1DLayer.class,
                Subsampling3DLayer.class,
                Convolution1DLayer.class,
                Convolution3DLayer.class,
                ConvolutionLayer.class,
                Upsampling1D.class,
                Upsampling2D.class,
                Upsampling3D.class,
                Deconvolution2D.class,
                Deconvolution3D.class,
                CnnLossLayer.class,
                CenterLossOutputLayer.class,
                RnnOutputLayer.class,
                OutputLayer.class,
                LastTimeStep.class,
                Cropping1DLayer.class,
                Cropping2DLayer.class,
                Cropping3DLayer.class,
                Cropping1D.class,
                Cropping2D.class,
                Cropping3D.class,
                SeparableConvolution2DLayer.class,
                ZeroPadding1DLayer.class,
                ZeroPadding3DLayer.class,
                ZeroPaddingLayer.class,
                SpaceToBatch.class,
                SpaceToDepth.class,
                BatchToSpace.class,
                DepthToSpace.class,
                DepthwiseConvolution2D.class);
    }


    static {
        try {
            loadConfigClasses();
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }


}
