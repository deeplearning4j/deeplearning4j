/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.codegen.ops

import org.nd4j.codegen.api.AtLeast
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*
import org.nd4j.codegen.api.DataType.*
import org.nd4j.codegen.api.Exactly

fun SDCNN() =  Namespace("CNN"){
    val namespaceJavaPackage = "org.nd4j.linalg.api.ops.impl.layers.convolution"

    val dataFormat = Mixin("dataFormat"){
        Arg(ENUM, "dataFormat") { possibleValues = listOf("NCHW", "NHWC"); description = "Data format: \"NCHW\" or \"NHWC\"" }
    }


    val conv1DConfig = Config("Conv1DConfig"){
        Arg(LONG, "k"){ description = "Kernel"; defaultValue=-1L}
        Arg(LONG, "s"){ description = "stride"; defaultValue=1}
        Arg(LONG, "p"){ description = "padding"; defaultValue=0}
        Arg(LONG, "d"){ description = "dilation"; defaultValue=1}
        Arg(BOOL, "isSameMode"){ description = "Same mode"; defaultValue=true}
        Arg(STRING, "dataFormat"){ description = "Data format"; defaultValue="NCW"}
        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv1DConfig"
    }

    val conv2DConfig = Config("Conv2DConfig"){
        Arg(LONG, "kH"){ description = "Kernel height"; defaultValue=-1L}
        Arg(LONG, "kW"){ description = "Kernel width"; defaultValue=-1L}
        Arg(LONG, "sH"){ description = "Stride along height dimension"; defaultValue=1};
        Arg(LONG, "sW"){ description = "Stride along width dimension"; defaultValue=1};
        Arg(LONG, "pH"){ description = "Padding along height dimension"; defaultValue=0};
        Arg(LONG, "pW"){ description = "Padding along width dimension"; defaultValue=0};
        Arg(LONG, "dH"){ description = "Dilation along height dimension"; defaultValue=1};
        Arg(LONG, "dW"){ description = "Dilation along width dimension"; defaultValue=1};
        Arg(BOOL, "isSameMode"){ description = "Same mode"; defaultValue=true}
        Arg(STRING, "dataFormat"){ description = "Data format"; defaultValue="NCHW"}
        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig"
    }

    val conv3DConfig = Config("Conv3DConfig"){
        Arg(LONG, "kD"){ description = "Kernel depth"; defaultValue=-1}
        Arg(LONG, "kW"){ description = "Kernel width"; defaultValue=-1}
        Arg(LONG, "kH"){ description = "Kernel height"; defaultValue=-1};
        Arg(LONG, "sD"){ description = "Stride depth"; defaultValue=1};
        Arg(LONG, "sW"){ description = "Stride width"; defaultValue=1};
        Arg(LONG, "sH"){ description = "Stride height"; defaultValue=1};
        Arg(LONG, "pD"){ description = "Padding depth"; defaultValue=0};
        Arg(LONG, "pW"){ description = "Padding width"; defaultValue=0};
        Arg(LONG, "pH"){ description = "Padding height"; defaultValue=0};
        Arg(LONG, "dD"){ description = "Dilation depth"; defaultValue=1};
        Arg(LONG, "dW"){ description = "Dilation width"; defaultValue=1};
        Arg(LONG, "dH"){ description = "Dilation height"; defaultValue=1};
        Arg(BOOL, "biasUsed"){ description = "biasUsed"; defaultValue=false}
        Arg(BOOL, "isSameMode"){ description = "Same mode"; defaultValue=true}
        Arg(STRING, "dataFormat"){ description = "Data format"; defaultValue="NDHWC"}
        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig"
    }


    val deconv2DConfig = Config("DeConv2DConfig"){
        Arg(LONG, "kH"){ description = "Kernel height"; defaultValue=-1L}
        Arg(LONG, "kW"){ description = "Kernel width"; defaultValue=-1L}
        Arg(LONG, "sH"){ description = "Stride along height dimension"; defaultValue=1L};
        Arg(LONG, "sW"){ description = "Stride along width dimension"; defaultValue=1L};
        Arg(LONG, "pH"){ description = "Padding along height dimension"; defaultValue=0};
        Arg(LONG, "pW"){ description = "Padding along width dimension"; defaultValue=0};
        Arg(LONG, "dH"){ description = "Dilation along height dimension"; defaultValue=1L};
        Arg(LONG, "dW"){ description = "Dilation along width dimension"; defaultValue=1L};
        Arg(BOOL, "isSameMode"){ description = "Same mode"; defaultValue=false}
        Arg(STRING, "dataFormat"){ description = "Data format"; defaultValue="NCHW"}
        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv2DConfig"
    }


    val deconv3DConfig = Config("DeConv3DConfig"){
        Arg(LONG, "kD"){ description = "Kernel depth"; defaultValue=-1L}
        Arg(LONG, "kW"){ description = "Kernel width"; defaultValue=-1L}
        Arg(LONG, "kH"){ description = "Kernel height"; defaultValue=-1L};
        Arg(LONG, "sD"){ description = "Stride depth"; defaultValue=1L};
        Arg(LONG, "sW"){ description = "Stride width"; defaultValue=1L};
        Arg(LONG, "sH"){ description = "Stride height"; defaultValue=1L};
        Arg(LONG, "pD"){ description = "Padding depth"; defaultValue=0};
        Arg(LONG, "pW"){ description = "Padding width"; defaultValue=0};
        Arg(LONG, "pH"){ description = "Padding height"; defaultValue=0};
        Arg(LONG, "dD"){ description = "Dilation depth"; defaultValue=1L};
        Arg(LONG, "dW"){ description = "Dilation width"; defaultValue=1L};
        Arg(LONG, "dH"){ description = "Dilation height"; defaultValue=1L};
        Arg(BOOL, "isSameMode"){ description = "Same mode"; defaultValue=false}
        Arg(STRING, "dataFormat"){ description = "Data format"; defaultValue="NCDHW"}
        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv3DConfig"
    }




    val pooling2DConfig = Config("Pooling2DConfig"){
        Arg(LONG, "kH"){ description = "Kernel height"; defaultValue=-1}
        Arg(LONG, "kW"){ description = "Kernel width"; defaultValue=-1}
        Arg(LONG, "sH"){ description = "Stride along height dimension"; defaultValue=1};
        Arg(LONG, "sW"){ description = "Stride along width dimension"; defaultValue=1};
        Arg(LONG, "pH"){ description = "Padding along height dimension"; defaultValue=0};
        Arg(LONG, "pW"){ description = "Padding along width dimension"; defaultValue=0};
        Arg(LONG, "dH"){ description = "Dilation along height dimension"; defaultValue=1};
        Arg(LONG, "dW"){ description = "Dilation along width dimension"; defaultValue=1};
        Arg(BOOL, "isSameMode"){ description = "Same mode"; defaultValue=true}
        Arg(STRING, "dataFormat"){ description = "Data format"; defaultValue="nchw"}
        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig"
    }

    val pooling3DConfig = Config("Pooling3DConfig"){
        Arg(LONG, "kD"){ description = "Kernel depth"; defaultValue=-1}
        Arg(LONG, "kW"){ description = "Kernel width"; defaultValue=-1}
        Arg(LONG, "kH"){ description = "Kernel height"; defaultValue=-1};
        Arg(LONG, "sD"){ description = "Stride depth"; defaultValue=1};
        Arg(LONG, "sW"){ description = "Stride width"; defaultValue=1};
        Arg(LONG, "sH"){ description = "Stride height"; defaultValue=1};
        Arg(LONG, "pD"){ description = "Padding depth"; defaultValue=0};
        Arg(LONG, "pW"){ description = "Padding width"; defaultValue=0};
        Arg(LONG, "pH"){ description = "Padding height"; defaultValue=0};
        Arg(LONG, "dD"){ description = "Dilation depth"; defaultValue=1};
        Arg(LONG, "dW"){ description = "Dilation width"; defaultValue=1};
        Arg(LONG, "dH"){ description = "Dilation height"; defaultValue=1};
        Arg(BOOL, "isSameMode"){ description = "Same mode"; defaultValue=true}
        Arg(STRING, "dataFormat"){ description = "Data format"; defaultValue="NCDHW"}
        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling3DConfig"
    }


    val LocalResponseNormalizationConfig = Config("LocalResponseNormalizationConfig"){
        Arg(NUMERIC, "alpha"){ description = "alpha"; defaultValue=1}
        Arg(NUMERIC, "beta"){ description = "beta"; defaultValue=0.5}
        Arg(NUMERIC, "bias"){ description = "bias"; defaultValue=1}
        Arg(INT, "depth"){ description = "depth"; defaultValue=5}
        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.convolution.config.LocalResponseNormalizationConfig"


    }




    Op("avgPooling2d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "AvgPooling2D"
        Input(NUMERIC, "input") { description = "the input to average pooling 2d operation - 4d CNN (image) activations in NCHW format (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])" }
        useConfig(pooling2DConfig)

        Output(NUMERIC, "output"){ description = "Result after applying average pooling on the input" }

        Doc(Language.ANY, DocScope.ALL){
            """
             2D Convolution layer operation - average pooling 2d
            """.trimIndent()
        }
    }

    Op("avgPooling3d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "AvgPooling3D"
        Input(NUMERIC, "input") {description = "the input to average pooling 3d operation - 5d activations in NCDHW format (shape [minibatch, channels, depth, height, width]) or NDHWC format (shape [minibatch, depth, height, width, channels])" }
        useConfig(pooling3DConfig)

        Output(NUMERIC, "output"){ description = "after applying average pooling on the input" }

        Doc(Language.ANY, DocScope.ALL){
        """
         3D convolution layer operation - average pooling 3d 
        """.trimIndent()
        }
    }

    Op("batchToSpace") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "BatchToSpace"
        Input(NUMERIC, "x") { description = "Input variable. 4d input" }
        Arg(INT, "blocks") { count=Exactly(2); description = "Block size, in the height/width dimension" }
        Arg(INT, "croppingTop") { count=Exactly(2)}
        Arg(INT, "croppingBottom") { count=Exactly(2)}
        Output(NUMERIC, "output"){ description = "Output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Convolution 2d layer batch to space operation on 4d input.
             Reduces input batch dimension by rearranging data into a larger spatial dimensions
            """.trimIndent()
        }
    }

    Op("col2Im") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "Col2Im"

        Input(NUMERIC, "in") { description = "Input - rank 6 input with shape [minibatch, inputChannels, kernelHeight, kernelWidth, outputHeight, outputWidth]" }
        useConfig(conv2DConfig)

        Output(NUMERIC, "output"){ description = "Col2Im output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """
             col2im operation for use in 2D convolution operations. Outputs a 4d array with shape
             [minibatch, inputChannels, height, width]
            """.trimIndent()
        }
    }



    Op("conv1d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "Conv1D"
        Input(NUMERIC, "input") { description = "the inputs to conv1d" }
        Input(NUMERIC, "weights") { description = "weights for conv1d op - rank 3 array with shape [kernelSize, inputChannels, outputChannels]" }
        Input(NUMERIC, "bias") { description = "bias for conv1d op - rank 1 array with shape [outputChannels]. May be null."; defaultValue=null }
        useConfig(conv1DConfig)

        Output(NUMERIC, "output"){ description = "result of conv1d op" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Conv1d operation.
            """.trimIndent()
        }
    }



    Op("conv2d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "Conv2D"
        Input(NUMERIC, "layerInput") { description = "the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format" }
        Input(NUMERIC, "weights") { description = "Weights for the convolution operation. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, outputChannels]" }
        Input(NUMERIC, "bias") { description = "Optional 1D bias array with shape [outputChannels]. May be null."; defaultValue=null }
        useConfig(conv2DConfig)

        Output(NUMERIC, "output"){ description = "result of conv2d op" }

        Doc(Language.ANY, DocScope.ALL){
            """
             2D Convolution operation with optional bias
            """.trimIndent()
        }
    }




    Op("conv3d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "Conv3D"
        Input(NUMERIC, "input") { description = "the input to average pooling 3d operation - 5d activations in NCDHW format (shape [minibatch, channels, depth, height, width]) or NDHWC format (shape [minibatch, depth, height, width, channels])" }
        Input(NUMERIC, "weights") { description = " Weights for conv3d. Rank 5 with shape [kernelDepth, kernelHeight, kernelWidth, inputChannels, outputChannels]." }
        Input(NUMERIC, "bias") { description = " Optional 1D bias array with shape [outputChannels]. May be null."; defaultValue=null }
        useConfig(conv3DConfig)

        Output(NUMERIC, "output"){ description = "Conv3d output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Convolution 3D operation with optional bias 
            """.trimIndent()
        }
    }



    Op("deconv2d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "DeConv2D"
        Input(NUMERIC, "layerInput") { description = "the input to deconvolution 2d operation - 4d CNN (image) activations in NCHW format (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])" }
        Input(NUMERIC, "weights") { description = "Weights for the 2d deconvolution operation. 4 dimensions with format [inputChannels, outputChannels, kernelHeight, kernelWidth]" }
        Input(NUMERIC, "bias") { description = "Optional 1D bias array with shape [outputChannels]. May be null."; defaultValue=null }
        useConfig(deconv2DConfig)
        Output(NUMERIC, "output"){ description = "result of deconv2d op" }

        Doc(Language.ANY, DocScope.ALL){
            """
             2D deconvolution operation with optional bias
            """.trimIndent()
        }
    }





    Op("deconv3d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "DeConv3D"
        Input(NUMERIC, "input") { description = "Input array - shape [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)" }
        Input(NUMERIC, "weights") { description = "Weights array - shape [kD, kH, kW, oC, iC]" }
        Input(NUMERIC, "bias") { description = "Bias array - optional, may be null. If non-null, must have shape [outputChannels]"; defaultValue=null }
        useConfig(deconv3DConfig)

        Output(NUMERIC, "output"){ description = "result of 3D CNN deconvolution operation" }

        Doc(Language.ANY, DocScope.ALL){
            """
             3D CNN deconvolution operation with or without optional bias
            """.trimIndent()
        }
    }

    Op("depthToSpace") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "DepthToSpace"
        Input(NUMERIC, "x") { description = "the input to depth to space pooling 2d operation - 4d activations in NCHW format (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])" }
        Arg(INT, "blockSize") { description = "Block size, in the height/width dimension" }
        useMixin(dataFormat)
        Output(NUMERIC, "output"){ description = "Output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Convolution 2d layer batch to space operation on 4d input.<br>
             Reduces input channels dimension by rearranging data into a larger spatial dimensions<br>
             Example: if input has shape [mb, 8, 2, 2] and block size is 2, then output size is [mb, 8/(2*2), 2*2, 2*2]
             = [mb, 2, 4, 4]
            """.trimIndent()
        }
    }


    Op("depthWiseConv2d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "DepthwiseConv2D"
        Input(NUMERIC, "layerInput") { description = "the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format" }
        Input(NUMERIC, "depthWeights") { description = "Depth-wise conv2d weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier]" }
        Input(NUMERIC, "bias") { description = "Optional 1D bias array with shape [outputChannels]. May be null."; defaultValue=null }
        useConfig(conv2DConfig)

        Output(NUMERIC, "output"){ description = "result of depthwise conv2d op" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Depth-wise 2D convolution operation with optional bias 
            """.trimIndent()
        }
    }



    Op("dilation2D") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "Dilation2D"
        Input(NUMERIC, "df") { description = "" }
        Input(NUMERIC, "weights") { description = "df" }
        Arg(INT, "strides") { count = Exactly(2); description = "weights" }
        Arg(INT, "rates") {count = Exactly(2); description = "strides" }
        Arg(BOOL, "isSameMode") { description = "isSameMode" }

        Output(NUMERIC, "output"){ description = "Computed the grayscale dilation of 4-D input and 3-D filters tensors." }

        Doc(Language.ANY, DocScope.ALL){
            """
             TODO doc string
            """.trimIndent()
        }
    }

    Op("extractImagePatches") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.image"
        javaOpClass = "ExtractImagePatches"
        Input(NUMERIC, "input") { description = "Input array. Must be rank 4, with shape [minibatch, height, width, channels]" }
        Arg(INT, "kH") { description = "Kernel height" }
        Arg(INT, "kW") { description = "Kernel width" }
        Arg(INT, "sH") { description = "Stride height" }
        Arg(INT, "sW") { description = "Stride width" }
        Arg(INT, "rH") { description = "Rate height" }
        Arg(INT, "rW") { description = "Rate width" }
        Arg(BOOL, "sameMode") { description = "If true: use same mode padding. If false" }

        Output(NUMERIC, "output"){ description = "The result is a 4D tensor which is indexed by batch, row, and column." }

        Doc(Language.ANY, DocScope.ALL){
            """
             Extract image patches 
            """.trimIndent()
        }
    }

    Op("im2Col") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "Im2col"
        Input(NUMERIC, "in") { description = "Input - rank 4 input with shape [minibatch, inputChannels, height, width]" }
        useConfig(conv2DConfig)

        Output(NUMERIC, "output"){ description = "Im2Col output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """
             im2col operation for use in 2D convolution operations. Outputs a 6d array with shape
             [minibatch, inputChannels, kernelHeight, kernelWidth, outputHeight, outputWidth]   
            """.trimIndent()
        }
    }

    Op("localResponseNormalization") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "LocalResponseNormalization"
        Input(NUMERIC, "input") { description = "the inputs to lrn" }
        useConfig(LocalResponseNormalizationConfig)

        Output(NUMERIC, "output"){ description = "Result after Local Response Normalization"}

        Doc(Language.ANY, DocScope.ALL){
            """
             2D convolution layer operation - local response normalization
            """.trimIndent()
        }
    }

    Op("maxPooling2d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "MaxPooling2D"
        Input(NUMERIC, "input") { description = "the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])" }
        useConfig(pooling2DConfig)

        Output(NUMERIC, "output"){ description = "Result after applying max pooling on the input" }

        Doc(Language.ANY, DocScope.ALL){
            """
             2D Convolution layer operation - max pooling 2d 
            """.trimIndent()
        }
    }

    Op("maxPoolWithArgmax") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "MaxPoolWithArgmax"
        Input(NUMERIC, "input") { description = "the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])" }
        useConfig(pooling2DConfig)

        Output(NUMERIC, "output"){ description = "Result after applying max pooling on the input" }
        Output(NUMERIC, "indexes"){ description = "Argmax array" }

        Doc(Language.ANY, DocScope.ALL){
            """
             2D Convolution layer operation - Max pooling on the input and outputs both max values and indices 
            """.trimIndent()
        }
    }

    Op("maxPooling3d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "MaxPooling3D"
        Input(NUMERIC, "input") { description = "the input to average pooling 3d operation - 5d activations in NCDHW format (shape [minibatch, channels, depth, height, width]) or NDHWC format (shape [minibatch, depth, height, width, channels])" }
        useConfig(pooling3DConfig)

        Output(NUMERIC, "output"){ description = "Result after applying max pooling on the input" }

        Doc(Language.ANY, DocScope.ALL){
            """
             3D convolution layer operation - max pooling 3d operation.
            """.trimIndent()
        }
    }



    Op("separableConv2d") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.layers.convolution"
        javaOpClass = "SConv2D"
        Input(NUMERIC, "layerInput") { description = "the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])" }
        Input(NUMERIC, "depthWeights") { description = "Separable conv2d depth weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier]" }
        Input(NUMERIC, "pointWeights") { description = "Point weights, rank 4 with format [1, 1, inputChannels*depthMultiplier, outputChannels]. May be null" }
        Input(NUMERIC, "bias") { description = "Optional bias, rank 1 with shape [outputChannels]. May be null."; defaultValue=null}
        useConfig(conv2DConfig)

        Output(NUMERIC, "output"){ description = "result of separable convolution 2d operation" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Separable 2D convolution operation with optional bias 
            """.trimIndent()
        }
    }



    Op("spaceToBatch") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "SpaceToBatch"
        Input(NUMERIC, "x") { description = "Input variable. 4d input" }
        Arg(INT, "blocks") { count = Exactly(2); description = "Block size, in the height/width dimension" }
        Arg(INT, "paddingTop") {count = Exactly(2); description = "Optional 2d int[] array for padding the result: values [[pad top, pad bottom], [pad left, pad right]]" }
        Arg(INT, "paddingBottom") {count = Exactly(2); description = "Optional 2d int[] array for padding the result: values [[pad top, pad bottom], [pad left, pad right]]" }

        Output(NUMERIC, "output"){ description = "Output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Convolution 2d layer space to batch operation on 4d input.
             Increases input batch dimension by rearranging data from spatial dimensions into batch dimension 
            """.trimIndent()
        }
    }

    Op("spaceToDepth") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "the input to depth to space pooling 2d operation - 4d activations in NCHW format (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])" }
        Arg(INT, "blockSize") { description = " Block size, in the height/width dimension" }
        useMixin(dataFormat)
        Output(NUMERIC, "output"){ description = "Output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Convolution 2d layer space to depth operation on 4d input.<br>
             Increases input channels (reduced spatial dimensions) by rearranging data into a larger channels dimension<br>
             Example: if input has shape [mb, 2, 4, 4] and block size is 2, then output size is [mb, 8/(2*2), 2*2, 2*2]
             = [mb, 2, 4, 4] 
            """.trimIndent()
        }
    }

    Op("upsampling2d") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "input") { description = "Input in NCHW format" }
        Arg(INT, "scale") { description = "The scale for both height and width dimensions." }

        Output(NUMERIC, "output"){ description = "Upsampled input"}

        Doc(Language.ANY, DocScope.ALL){
            """
             Upsampling layer for 2D inputs.
             scale is used for both height and width dimensions. 
            """.trimIndent()
        }
    }

    Op("upsampling2d") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "input") { description = "Input in NCHW format" }
        Arg(INT, "scaleH") { description = "Scale to upsample in height dimension" }
        Arg(INT, "scaleW") { description = "Scale to upsample in width dimension" }
        Arg(BOOL ,"nchw") { description = "If true: input is in NCHW (minibatch, channels, height, width) format. False: NHWC format" }


        Output(NUMERIC, "output"){ description = "Upsampled input" }

        Doc(Language.ANY, DocScope.ALL){
            """
             2D Convolution layer operation - Upsampling 2d 
            """.trimIndent()
        }
    }

    Op("upsampling3d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "Upsampling3d"
        Input(NUMERIC, "input") { description = "Input in NCHW format" }
        Arg(BOOL ,"ncdhw") { description = "If true: input is in NCDHW (minibatch, channels, depth, height, width) format. False: NDHWC format" }
        Arg(INT, "scaleD") { description = "Scale to upsample in depth dimension" }
        Arg(INT, "scaleH") { description = "Scale to upsample in height dimension" }
        Arg(INT, "scaleW") { description = "Scale to upsample in width dimension" }


        Output(NUMERIC, "output"){ description = "Upsampled input" }

        Doc(Language.ANY, DocScope.ALL){
            """
             3D Convolution layer operation - Upsampling 3d 
            """.trimIndent()
        }
    }
}