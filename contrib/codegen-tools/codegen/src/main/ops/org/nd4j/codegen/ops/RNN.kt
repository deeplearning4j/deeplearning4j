package org.nd4j.codegen.ops

import org.nd4j.codegen.api.AtLeast
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*
import org.nd4j.codegen.api.DataType.*

fun SDRNN() = Namespace("RNN") {


    val LSTMConfiguration = Config("LSTMConfiguration") {

        Arg(ENUM, "RnnDataFormat") {
            possibleValues = listOf("TNS", "NST", "NTS"); description = " The data format of the input. Input shape depends on data format (in config):<br>\n" +
                " TNS -> [timeSteps, batchSize, inSize]<br>\n" +
                " NST -> [batchSize, inSize, timeSteps]<br>\n" +
                " NTS -> [batchSize, timeSteps, inSize]<br>"
        }


        Arg(BOOL, "peepHole") { description = "Whether to provide peephole connections"; }
        Arg(NUMERIC, "forgetBias") { description = "The bias added to forget gates in order to reduce the scale of forgetting in the beginning of the training."; }
        Arg(NUMERIC, "clippingCellValue") { description = "The bias added to forget gates in order to reduce the scale of forgetting in the beginning of the training."; }

        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMConfiguration"
    }


    val LSTMLayerConfig = Config("LSTMLayerConfig") {

        Arg(ENUM, "LSTMDataFormat") {
            possibleValues = listOf("TNS", "NST", "NTS", "T2NS");
            description = "for unidirectional:" +
                    "  TNS: shape [timeLength, numExamples, inOutSize] - sometimes referred to as \"time major\"<br>\n" +
                    "  NST: shape [numExamples, inOutSize, timeLength]<br>\n" +
                    "  NTS: shape [numExamples, timeLength, inOutSize] - TF \"time_major=false\" layout<br>" +
                    " for bidirectional:\n" +
                    "   T2NS: 3 = [timeLength, 2, numExamples, inOutSize] (for ONNX)"
        }


        Arg(ENUM, "LSTMDirectionMode") {
            possibleValues = listOf("FWD", "BWD", "BIDIR_SUM", "BIDIR_CONCAT", "BIDIR_EXTRA_DIM"); description = "direction <br>\n" +
                " FWD: 0 = fwd\n" +
                " BWD: 1 = bwd\n" +
                " BIDIR_SUM: 2 = bidirectional sum\n" +
                " BIDIR_CONCAT: 3 = bidirectional concat\n" +
                " BIDIR_EXTRA_DIM: 4 = bidirectional extra output dim (in conjunction with format dataFormat = 3)"
        }

        Arg(ENUM, "gateAct") {
            possibleValues = listOf("TANH",
                    "RELU",
                    "SIGMOID",
                    "AFFINE",
                    "LEAKY_RELU",
                    "THRESHHOLD_RELU",
                    "SCALED_TAHN",
                    "HARD_SIGMOID",
                    "ELU",
                    "SOFTSIGN",
                    "SOFTPLUS"); description = "Activations"
        }


        Arg(ENUM, "cellAct") {
            possibleValues = listOf("TANH",
                    "RELU",
                    "SIGMOID",
                    "AFFINE",
                    "LEAKY_RELU",
                    "THRESHHOLD_RELU",
                    "SCALED_TAHN",
                    "HARD_SIGMOID",
                    "ELU",
                    "SOFTSIGN",
                    "SOFTPLUS"); description = "Activations"
        }


        Arg(ENUM, "outAct") {
            possibleValues = listOf("TANH",
                    "RELU",
                    "SIGMOID",
                    "AFFINE",
                    "LEAKY_RELU",
                    "THRESHHOLD_RELU",
                    "SCALED_TAHN",
                    "HARD_SIGMOID",
                    "ELU",
                    "SOFTSIGN",
                    "SOFTPLUS"); description = "Activations"
        }


        Arg(BOOL, "retFullSequence") { description = "indicates whether to return whole time sequence h {h_0, h_1, ... , h_sL-1}"; defaultValue = true }
        Arg(BOOL, "retLastH") {
            description = "indicates whether to return output at last time step only,\n" +
                    " in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)"; defaultValue = false
        }
        Arg(BOOL, "retLastC") {
            description = "indicates whether to return cells state at last time step only,\n" +
                    " in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)"; defaultValue = false
        }
        Arg(NUMERIC, "cellClip") { description = "Cell clipping value, if it = 0 then do not apply clipping"; defaultValue = 0.0}

        Arg(NUMERIC, "gateAlpha") {defaultValue=0.0}
        Arg(NUMERIC, "gateBeta") {defaultValue=0.0}
        Arg(NUMERIC, "cellAlpha") {defaultValue=0.0}
        Arg(NUMERIC, "cellBeta") {defaultValue=0.0}
        Arg(NUMERIC, "outAlpha") {defaultValue=0.0}
        Arg(NUMERIC, "outBeta") {defaultValue=0.0}


       javaClassOverride =  "org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMLayerConfig"
    }


    val GRUWeights = Config("GRUWeights") {
        Input(NUMERIC, "ruWeight")
        Input(NUMERIC, "cWeight")
        Input(NUMERIC, "ruBias")
        Input(NUMERIC, "cBias")
        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.GRUWeights"
    }

    val SRUWeights = Config("SRUWeights") {
        Input(NUMERIC, "weights")
        Input(NUMERIC, "bias")
        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.SRUWeights"
    }

    val LSTMWeights = Config("LSTMWeights") {
        Input(NUMERIC, "ruWeight")
        Input(NUMERIC, "inputPeepholeWeights")
        Input(NUMERIC, "forgetPeepholeWeights")
        Input(NUMERIC, "outputPeepholeWeights")
        Input(NUMERIC, "bias")

        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMWeights"
    }

    val LSTMLayerWeights = Config("LSTMLayerWeights") {
        Input(NUMERIC, "inputWeights") {description="input weights Wx:\n" +
                " 1) shapes `[nIn, 4*nOut]` for FWD,BWD " +
                " 2) shapes `[2, nIn, 4*nOut]` BIDIR_SUM, BIDIR_CONCAT and BIDIR_EXTRA_DIM"}
        Input(NUMERIC, "recurrentWeights") {description="recurrent weights Wr:\n" +
                " 1) shapes `[nIn, 4*nOut]` for FWD, BWD " +
                " 2) shapes `[2, nIn, 4*nOut]` BIDIR_SUM, BIDIR_CONCAT and BIDIR_EXTRA_DIM"}
        Input(NUMERIC, "biases") {description="biases\n"+
                " 1) shapes `[4*nOut]` for FWD, BWD " +
                " 2) shapes `[2, 4*nOut]` for BIDIR_SUM, BIDIR_CONCAT and BIDIR_EXTRA_DIM"
                  defaultValue=null}
        Input(NUMERIC, "peepholeWeights") {description="peephole weights Wp:\n" +
                "  1) `[3*nOut]`    when directionMode <  2\n" +
                "  2) `[2, 3*nOut]`  when directionMode >= 2"; defaultValue=null}


        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMLayerWeights"
    }


    val namespaceJavaPackage = "org.nd4j.linalg.api.ops.impl.layers.recurrent"
    Op("gruCell") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "GRUCell"
        Input(NUMERIC, "x") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "hLast") { description = "Output of the previous cell/time step, with shape [batchSize, numUnits]" }
        useConfig(GRUWeights)
        Output(NUMERIC, "r") { description = "Reset gate output" }
        Output(NUMERIC, "u") { description = "Update gate output" }
        Output(NUMERIC, "c") { description = "Cell gate output" }
        Output(NUMERIC, "h") { description = "Cell output" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            The GRU cell.  Does a single time step operation
            """.trimIndent()
        }
    }

    Op("gru") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "GRU"
        Input(NUMERIC, "x") { description = "input [time, bS, nIn]" }
        Input(NUMERIC, "hLast") { description = "initial cell output (at time step = 0) [bS, nOut]" }
        Input(NUMERIC, "Wx") { description = "input-to-hidden  weights, [nIn, 3*nOut]" }
        Input(NUMERIC, "Wh") { description = "hidden-to-hidden weights, [nOut, 3*nOut]" }
        Input(NUMERIC, "biases") { description = "biases, [3*nOut]" }

        Output(NUMERIC, "h") { description = "cell outputs [time, bS, nOut], that is per each time step" }



        Doc(Language.ANY, DocScope.ALL) {
            """
            The GRU operation. Gated Recurrent Unit - Cho et al. 2014.


            """.trimIndent()
        }
    }






    Op("lstmCell") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "LSTMBlockCell"
        Input(NUMERIC, "x") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "cLast") { description = "Previous cell state, with shape [batchSize, numUnits]" }
        Input(NUMERIC, "yLast") { description = "revious cell output, with shape [batchSize, numUnits]" }
        useConfig(LSTMWeights)
        useConfig(LSTMConfiguration)

        Output(NUMERIC, "i") { description = "Output - input modulation gate activations [batchSize, numUnits]." }
        Output(NUMERIC, "c") { description = "Output - Activations, cell state (pre tanh) [batchSize, numUnits]." }
        Output(NUMERIC, "f") { description = "Output - forget gate activations [batchSize, numUnits]." }
        Output(NUMERIC, "o") { description = "Output - output gate activations [batchSize, numUnits]." }
        Output(NUMERIC, "z") { description = "Output - input gate activations [batchSize, numUnits]." }
        Output(NUMERIC, "h") { description = "Cell state, post tanh [batchSize, numUnits]." }
        Output(NUMERIC, "y") { description = "Current cell output [batchSize, numUnits]." }

        Doc(Language.ANY, DocScope.ALL) {
            """
            The LSTM cell.  Does a single time step operation.
            """.trimIndent()
        }
    }



    Op("lstmblock") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "LSTMBlock"
        Input(NUMERIC, "maxTSLength") {defaultValue=null}
        Input(NUMERIC, "x") { description = " Input, with shape dependent on the data format (in config)." }
        Input(NUMERIC, "cLast") { description = "Previous/initial cell state, with shape [batchSize, numUnits]" ; defaultValue=null}
        Input(NUMERIC, "yLast") { description = "Previous/initial cell output, with shape [batchSize, numUnits]" ; defaultValue=null }
        useConfig(LSTMWeights)
        useConfig(LSTMConfiguration)

        Output(NUMERIC, "output") { description = "The layer's outputs." }

        Doc(Language.ANY, DocScope.ALL) {
            """
             The LSTM block
            """.trimIndent()
        }
    }



    Op("lstmLayer") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "LSTMLayer"
        Input(NUMERIC, "x") { description = " Input, with shape dependent on the data format (in config)." }
        Input(NUMERIC, "cLast") { description = "Previous/initial cell state, with shape [batchSize, numUnits]"; defaultValue=null }
        Input(NUMERIC, "yLast") { description = "Previous/initial cell output, with shape [batchSize, numUnits]"; defaultValue=null }
        Input(NUMERIC, "maxTSLength") { description = "maxTSLength with shape [batchSize]"; defaultValue=null }
        useConfig(LSTMLayerWeights)
        useConfig(LSTMLayerConfig)

        //TODO these are optional
        Output(NUMERIC, "output") { description = "The layer's outputs - full time series" }
        Output(NUMERIC, "yLast") { description = "The layer's outputs - last time step activations (yLast)" }
        Output(NUMERIC, "cLast") { description = "The layer's outputs - last time step cell state (cLast)" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Long Short-Term Memory layer - Hochreiter 1997.
             SUPPORTS following data formats:
             for unidirectional:
             TNS: shapes [timeLength, numExamples, inOutSize]
             NST: shapes [numExamples, inOutSize, timeLength]
             NTS: shapes [numExamples, timeLength, inOutSize]
             for bidirectional:
             T2NS: shapes [timeLength, 2, numExamples, inOutSize] (for ONNX)
             SUPPORTS following direction modes:
             FWD: forward
             BWD: backward
             BIDIR_SUM: bidirectional sum
             BIDIR_CONCAT: bidirectional concat
             BIDIR_EXTRA_DIM: bidirectional extra output dim (in conjunction with format dataFormat - T2NS)
             You may use different gate configurations:
             specify gate/cell/out aplha/beta and numbers of activations for gate/cell/out described in activations enum
             ("RELU","SIGMOID","AFFINE","LEAKY_RELU","THRESHHOLD_RELU","SCALED_TAHN","HARD_SIGMOID","ELU","SOFTSIGN","SOFTPLUS")
             Also this layer supports MKLDNN (DNNL) and cuDNN acceleration
            """.trimIndent()
        }
    }



    Op("sruCell") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "SRUCell"
        Input(NUMERIC, "x") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "cLast") { description = "Previous cell state, with shape [batchSize, inSize]" }
        useConfig(SRUWeights)

        Output(NUMERIC, "output") { description = "The cell's outputs." }

        Doc(Language.ANY, DocScope.ALL) {
            """
             The SRU layer.  Does a single time step operation.
            """.trimIndent()
        }
    }


    Op("sru") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "SRU"
        Input(NUMERIC, "x") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "initialC") { description = "Initial cell state, with shape [batchSize, inSize]" }
        Input(NUMERIC, "mask") { description = "An optional dropout mask, with shape [batchSize, inSize]"; defaultValue = null }

        useConfig(SRUWeights)

        Output(NUMERIC, "output") { description = "The cell's outputs.." }

        Doc(Language.ANY, DocScope.ALL) {
            """
             The SRU layer.  Does a single time step operation.
            """.trimIndent()
        }

    }
}




