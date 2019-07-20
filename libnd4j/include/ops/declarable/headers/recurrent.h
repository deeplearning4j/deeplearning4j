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

//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_HEADERS_RECURRENT_H
#define LIBND4J_HEADERS_RECURRENT_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
namespace ops  {

    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operation for Simple Recurrent Unit: "Training RNNs as Fast as CNNs" Tao Lei, Yu Zhang, Yoav Artzi
       *
       * Input arrays:
       *    0: input 3d tensor with shape [bS x K x N], N - number of time steps, bS - batch size, K - number of features
       *    1: 2d tensor of weights [3K x K]
       *    2: row of biases with twice length [1 x 2K]
       *    3: 2d tensor of previous cell state [bS x K]
       *    4: optional, 2d tensor of dropout mask [bS x K]
       *
       * Output arrays:
       *    0: 3d tensor of cell output [bS x K x N]
       *    1: 3d tensor of cell state [bS x K x N]
       */
        #if NOT_EXCLUDED(OP_sru)
        DECLARE_CUSTOM_OP(sru,   5, 2, false, 0, 0);
        #endif

    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operation for Simple Recurrent Unit (bidirectional case): "Training RNNs as Fast as CNNs" Tao Lei, Yu Zhang, Yoav Artzi
       *
       * Input arrays:
       *    0: input 3d tensor with shape [N x bS x 2K], N - number of time steps, bS - batch size, K - number of features
       *    1: 2d tensor of weights [2K x 6K]
       *    2: row of biases with twice length [1 x 4K]
       *    3: 2d tensor of previous cell state [bS x 2K]
       *    4: optional, 2d tensor of dropout mask [bS x 2K]
       *
       * Output arrays:
       *    0: 3d tensor of cell output [N x bS x 2K]
       *    1: 3d tensor of cell state [N x bS x 2K]
       */
        #if NOT_EXCLUDED(OP_sru_bi)
        DECLARE_CUSTOM_OP(sru_bi,      5, 2, true,  0, 0);
        #endif


    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operation for back propagation in Simple Recurrent Unit: "Training RNNs as Fast as CNNs" Tao Lei, Yu Zhang, Yoav Artzi
       *
       * Input arrays:
       *    0: input 3d tensor with shape [bS x K x N], N - number of time steps, bS - batch size, K - number of features
       *    1: 2d tensor of weights [3K x K]
       *    2: row of biases with twice length [1 x 2K]
       *    3: 2d tensor of previous cell state [bS x K]
       *    4: 3d tensor of cell state [bS x K x N]
       *    5: 2d tensor of cell state gradients [bS x K]
       *    6: 3d tensor of state output gradients [bS x K x N]
       *    7: optional, 2d tensor of dropout mask [bS x K]
       *
       * Output arrays:
       *    0: 3d tensor of input gradients [bS x K x N]
       *    1: 3d tensor of weights gradients [bS x 3K x K]
       *    2: 2d, row of biases gradients [1 x 2K]
       *    3: 2d, tensor of state gradients [bS x K]
       */
        #if NOT_EXCLUDED(OP_sru)
        DECLARE_CUSTOM_OP(sru_bp,      8, 4, true,  0, 0);
        #endif

    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operation for back propagation in Simple Recurrent Unit (bidirectional case): "Training RNNs as Fast as CNNs" Tao Lei, Yu Zhang, Yoav Artzi
       *
       * Input arrays:
       *    0: input 3d tensor with shape [N x bS x 2K], N - number of time steps, bS - batch size, K - number of features
       *    1: 2d tensor of weights [2K x 6K]
       *    2: row of biases with twice length [1 x 4K]
       *    3: 2d tensor of previous cell state [bS x 2K]
       *    4: 3d tensor of cell state [N x bS x 2K]
       *    5: 2d tensor of cell state gradients [bS x 2K]
       *    6: 3d tensor of state output gradients [N x bS x 2K]
       *    7: optional, 2d tensor of dropout mask [bS x 2K]
       *
       * Output arrays:
       *    0: 3d tensor of input gradients [N x bS x 2K]
       *    1: 3d tensor of weights gradients [N x 2K x 6K]
       *    2: 2d, row of biases gradients [1 x 4K]
       *    3: 2d, tensor of state gradients [bS x 2K]
       */
        #if NOT_EXCLUDED(OP_sru_bi)
        DECLARE_CUSTOM_OP(sru_bi_bp,   8, 4, true,  0, 0);
        #endif


    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operation for LSTM cell with peep hole connections:
       *    S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural Computation
       *    and
       *    https://research.google.com/pubs/archive/43905.pdf
       *    Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory recurrent neural network architectures for large scale acoustic modeling." INTERSPEECH, 2014.
       *
       * Input arrays:
       *    0: input with shape [batchSize x inSize], batchSize - batch size, inSize - number of features
       *    1: previous cell output [batchSize x numProj],  that is at previous time step t-1, in case of projection=false -> numProj=numUnits!!!
       *    2: previous cell state  [batchSize x numUnits], that is at previous time step t-1
       *    3: input-to-hidden  weights, [inSize  x 4*numUnits]
       *    4: hidden-to-hidden weights, [numProj x 4*numUnits]
       *    5: diagonal weights for peephole connections [3*numUnits]
       *    6: projection weights [numUnits x numProj]
       *    7: biases, [4*numUnits]
       *
       *  Input integer arguments:
       *    0: if not zero, provide peephole connections
       *    1: if not zero, then projection is performed, if zero then numProj==numUnits is mandatory!
       *
       *  Input float arguments:
       *    0: clipping value for cell state, if it is not equal to zero, then cell state is clipped
       *    1: clipping value for projected cell output, if it is not equal to zero, then projected cell output is clipped
       *    2: the bias added to forget gates in order to reduce the scale of forgetting in the beginning of the training
       *
       * Output arrays:
       *    0: current cell output [batchSize x numProj], that is at current time step t
       *    1: current cell state  [batchSize x numUnits], that is at current time step t
       */
        #if NOT_EXCLUDED(OP_lstmCell)
        DECLARE_CUSTOM_OP(lstmCell, 8, 2, false, 3, 2);
        #endif


    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operation for LSTM cell with optional peep hole connections:
       *    S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural Computation
       *    and
       *    https://research.google.com/pubs/archive/43905.pdf
       *    Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory recurrent neural network architectures for large scale acoustic modeling." INTERSPEECH, 2014.
	   * See also: https://arxiv.org/pdf/1503.04069.pdf
       *
       * Input arrays:
       *    0: input [bS, inSize] at time t
       *    1: previous cell state  [bS, numUnits], time t-1
       *    2: previous output [bS, numUnits], time t-1
       *    3: Weights - concatenated (input-to-hidden, hidden-to-hidden weights)  weights, [(inSize+numUnits), 4*numUnits]
       *    4: weights - cell peephole (t-1) connections to input modulation gate, [numUnits]
       *    5: weights - cell peephole (t-1) connections to forget gate, [numUnits]
       *    6: weights - cell peephole (t) connections to output gate, [numUnits]
       *    7: biases, shape [4*numUnits]
       *
       *  Input integer arguments:
       *    0: if not zero, provide peephole connections
       *
       *  Input float arguments:
       *    0: the bias added to forget gates in order to reduce the scale of forgetting in the beginning of the training
	   *    1: clipping value for cell state, if it is not equal to zero, then cell state is clipped
       *
       * Output arrays:
       *    0: i      - Input modulation gate activations [bS, numUnits]
       *    1: c (cs) - Cell state (pre tanh) [bs, numUnits] (cs)
       *    2: f      - Output - forget gate activations [bs, numUnits]
       *    3: o      - Output - output gate activations [bs, numUnits]
       *    4: z (ci) - Output - block input [bs, numUnits]
       *    5: h (co) - Cell state, post tanh [bs, numUnits]
       *    6: y (h)  - Current cell output [bS, numUnits], time t
       */
        #if NOT_EXCLUDED(OP_lstmBlockCell)
        DECLARE_CUSTOM_OP(lstmBlockCell, 8, 7, false, 2, 1);
        #endif

    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operation for LSTM layer with optional peep hole connections.
       * See lstmBlockCell for details. lstmBlockCell is used internally for computation.
       * This method expects as input (and returns as output) sequences in one of 3 formats, depending on the data format arg:
       * dataFormat = 0 -> TNS: shape [timeLength, numExamples, inOutSize] - sometimes referred to as "time major"
       * dataFormat = 1 -> NST: shape [numExamples, inOutSize, timeLength]
       * dataFormat = 2 -> NTS: shape [numExamples, timeLength, inOutSize] - TF "time_major=false" layout
       *
       *
       * Input arrays:
       *    0: max sequence length; long/int64 scalar
       *    1: input [seqLength, bS, inSize] at time t
       *    2: previous/initial cell state  [bS, numUnits]
       *    3: previous/initial output [bS, numUnits]
       *    4: Weights - concatenated (input-to-hidden, hidden-to-hidden weights)  weights, [(inSize+numUnits), 4*numUnits]
       *    5: weights - cell peephole (t-1) connections to input modulation gate, [numUnits]
       *    6: weights - cell peephole (t-1) connections to forget gate, [numUnits]
       *    7: weights - cell peephole (t) connections to output gate, [numUnits]
       *    8: biases, Shape [4*numUnits]
       *
       *  Input integer arguments:
       *    0: if not zero, provide peephole connections
       *    1: Data format - 0=TNS=[seqLen,mb,size]; 1=NST=[mb,size,seqLen]; 2=NTS=[mb,seqLen,size]
       *
       *  Input float arguments:
       *    0: the bias added to forget gates in order to reduce the scale of forgetting in the beginning of the training
       *    1: clipping value for cell state, if it is not equal to zero, then cell state is clipped
       *
       * Output arrays:
       *    0: i      - Input modulation gate activations, rank 3, shape as per dataFormat
       *    1: c (cs) - Cell state (pre tanh), rank 3, shape as per dataFormat
       *    2: f      - Output - forget gate activations, rank 3, shape as per dataFormat
       *    3: o      - Output - output gate activations, rank 3, shape as per dataFormat
       *    4: z (ci) - Output - block input, rank 3, shape as per dataFormat
       *    5: h (co) - Cell state, post tanh, rank 3, shape as per dataFormat
       *    6: y (h)  - Current cell output, rank 3, shape as per dataFormat
       */
        #if NOT_EXCLUDED(OP_lstmBlock)
        DECLARE_CUSTOM_OP(lstmBlock, 9, 7, false, 2, 2);
        #endif

    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operations for Simple Recurrent Unit cell: "Training RNNs as Fast as CNNs" Tao Lei, Yu Zhang, Yoav Artzi
       *
       * Input arrays:
       *    0: input with shape [batchSize x inSize], batchSize - batch size, inSize - number of features
       *    1: previous cell state [batchSize x inSize], that is at previous time step t-1
       *    2: weights [inSize x 3*inSize]
       *    3: biases [1 x 2*inSize]
       *
       * Output arrays:
       *    0: current cell output [batchSize x inSize], that is at current time step t
       *    1: current cell state  [batchSize x inSize], that is at current time step t
       */
        #if NOT_EXCLUDED(OP_sruCell)
        DECLARE_CUSTOM_OP(sruCell, 4, 2, false, 0, 0);
        #endif


    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of gated Recurrent Unit cell:
       *    Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio
       *    "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
       *
       * Input arrays:
       *    0: input with shape [batchSize x inSize], batchSize - batch size, inSize - number of features
       *    1: previous cell output [batchSize x numUnits],  that is at previous time step t-1
       *    2: RU weights - [(inSize+numUnits), 2*numUnits] - reset and update gates (input/recurrent weights)
       *    3: C weights - [(inSize+numUnits), numUnits] - cell gate (input/recurrent weights)
       *    4: reset and update biases, [2*numUnits] - reset and update gates
       *    5: cell biases, [numUnits]
       *
       * Output arrays:
       *    0: Reset gate output [bS, numUnits]
       *    1: Update gate output [bS, numUnits]
       *    2: Cell gate output [bS, numUnits]
       *    3: Current cell output [bS, numUnits]
       */
        #if NOT_EXCLUDED(OP_gruCell)
        DECLARE_CUSTOM_OP(gruCell, 6, 4, false, 0, 0);
        #endif

        #if NOT_EXCLUDED(OP_gruCell)
        DECLARE_CUSTOM_OP(gruCell_bp, 10, 6, false, 0, 0);
        #endif

    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operation "LSTM time sequences" with peep hole connections:
       *
       * Input arrays:
       *    0: input with shape [time x batchSize x inSize], time - number of time steps, batchSize - batch size, inSize - number of features
       *    1: initial cell output [batchSize x numProj],  that is at time step = 0, in case of projection=false -> numProj=numUnits!!!
       *    2: initial cell state  [batchSize x numUnits], that is at time step = 0
       *    3: input-to-hidden  weights, [inSize  x 4*numUnits]
       *    4: hidden-to-hidden weights, [numProj x 4*numUnits]
       *    5: diagonal weights for peephole connections [3*numUnits]
       *    6: projection weights [numUnits x numProj]
       *    7: biases, [4*numUnits]
       *
       *  Input integer arguments:
       *    0: if not zero, provide peephole connections
       *    1: if not zero, then projection is performed, if zero then numProj==numUnits is mandatory!
       *
       *  Input float arguments:
       *    0: clipping value for cell state, if it is not equal to zero, then cell state is clipped
       *    1: clipping value for projected cell output, if it is not equal to zero, then projected cell output is clipped
       *    2: the bias added to forget gates in order to reduce the scale of forgetting in the beginning of the training
       *
       * Output arrays:
       *    0: cell outputs [time x batchSize x numProj], that is per each time step
       *    1: cell states  [time x batchSize x numUnits], that is per each time step
       */
        #if NOT_EXCLUDED(OP_lstm)
        DECLARE_CUSTOM_OP(lstm, 8, 2, false, 3, 2);
        #endif

    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of gated Recurrent Unit:
       *
       * Input arrays:
       *    0: input with shape [time x batchSize x inSize], time - number of time steps, batchSize - batch size, inSize - number of features
       *    1: initial cell output [batchSize x numUnits],  that is at time step = 0
       *    2: input-to-hidden  weights, [inSize   x 3*numUnits]
       *    3: hidden-to-hidden weights, [numUnits x 3*numUnits]
       *    4: biases, [3*numUnits]
       *
       * Output arrays:
       *    0: cell outputs [time x batchSize x numUnits], that is per each time step
       */
        #if NOT_EXCLUDED(OP_gru)
        DECLARE_CUSTOM_OP(gru, 5, 1, false, 0, 0);
        #endif

    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operation "static RNN time sequences" with peep hole connections:
       *
       * Input arrays:
       *    0: input with shape [time x batchSize x inSize], time - number of time steps, batchSize - batch size, inSize - number of features
       *    1: input-to-hidden  weights, [inSize   x numUnits]
       *    2: hidden-to-hidden weights, [numUnits x numUnits]
       *    3: biases, [2*numUnits]
       *    4: (optional) initial cell output [batchSize x numUnits], that is at time step = 0
       *    5: (optional) vector with shape [batchSize] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this provides no calculations for time >= maxTimeStep
       *
       * Output arrays:
       *    0: cell outputs [time x batchSize x numUnits]
       *    1: cell final non-zero output [batchSize x numUnits]
       */
        DECLARE_CUSTOM_OP(static_rnn, 4, 2, false, 0, 0);

    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operation "static RNN time sequences" with peep hole connections:
       *
       * Input arrays:
       *    0: input with shape [time x batchSize x inSize] or [batchSize x time x numUnits], time - number of time steps, batchSize - batch size, inSize - number of features
       *    1: input-to-hidden  weights, [inSize   x numUnits]
       *    2: hidden-to-hidden weights, [numUnits x numUnits]
       *    3: biases, [2*numUnits]
       *    4: (optional) initial cell output [batchSize x numUnits], that is at time step = 0
       *    5: (optional) vector with shape [batchSize] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this provides no calculations for time >= maxTimeStep
       *
       *  Input integer arguments:
       *    0: (optional) timeMajor - if non zero then input shape is [time, batchSize, ...], else [batchSize, time, ...]
       *
       * Output arrays:
       *    0: cell outputs [time x batchSize x numUnits] or [batchSize x time x numUnits]
       *    1: cell final non-zero output [batchSize x numUnits]
       */
        DECLARE_CUSTOM_OP(dynamic_rnn, 4, 2, false, 0, 0);

    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operation "static RNN time sequences" with peep hole connections:
       *
       * Input arrays:
       *    0: input with shape [time x batchSize x inSize], time - number of time steps, batchSize - batch size, inSize - number of features
       *    1: input-to-hidden  weights for forward RNN, [inSize   x numUnitsFW]
       *    2: hidden-to-hidden weights for forward RNN, [numUnitsFW x numUnitsFW]
       *    3: biases for forward RNN, [2*numUnitsFW]
       *    4: input-to-hidden  weights for backward RNN, [inSize   x numUnitsBW]
       *    5: hidden-to-hidden weights for backward RNN, [numUnitsBW x numUnitsBW]
       *    6: biases for backward RNN, [2*numUnitsBW]
       *    7: (optional) initial cell output for forward RNN [batchSize x numUnitsFW], that is at time step = 0
       *    8: (optional) initial cell output for backward RNN [batchSize x numUnitsBW], that is at time step = 0
       *    9: (optional) vector with shape [batchSize] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this provides no calculations for time >= maxTimeStep
       *
       * Output arrays:
       *    0: cell outputs [time x batchSize x (numUnitsFW + numUnitsBW)]
       *    1: cell final non-zero output for forward RNN  [batchSize x numUnitsFW]
       *    2: cell final non-zero output for backward RNN [batchSize x numUnitsBW]
       */
        DECLARE_CUSTOM_OP(static_bidirectional_rnn, 7, 3, false, 0, 0);

    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operation "static RNN time sequences" with peep hole connections:
       *
       * Input arrays:
       *    0: input with shape [time x batchSize x inSize] or [batchSize x time x inSize], time - number of time steps, batchSize - batch size, inSize - number of features
       *    1: input-to-hidden  weights for forward RNN, [inSize   x numUnitsFW]
       *    2: hidden-to-hidden weights for forward RNN, [numUnitsFW x numUnitsFW]
       *    3: biases for forward RNN, [2*numUnitsFW]
       *    4: input-to-hidden  weights for backward RNN, [inSize   x numUnitsBW]
       *    5: hidden-to-hidden weights for backward RNN, [numUnitsBW x numUnitsBW]
       *    6: biases for backward RNN, [2*numUnitsBW]
       *    7: (optional) initial cell output for forward RNN [batchSize x numUnitsFW], that is at time step = 0
       *    8: (optional) initial cell output for backward RNN [batchSize x numUnitsBW], that is at time step = 0
       *    9: (optional) vector with shape [batchSize] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this provides no calculations for time >= maxTimeStep
       *
       *  Input integer arguments:
       *    0: (optional) timeMajor - if non zero then input shape is [time, batchSize, ...], else [batchSize, time, ...]
       *
       * Output arrays:
       *    0: cell outputs for forward  RNN [time x batchSize x numUnitsFW] or [batchSize x time x  numUnitsFW]
       *    1: cell outputs for backward RNN [time x batchSize x numUnitsBW] or [batchSize x time x  numUnitsBW]
       *    2: cell final non-zero output for forward  RNN [batchSize x numUnitsFW]
       *    3: cell final non-zero output for backward RNN [batchSize x numUnitsBW]
       */
        DECLARE_CUSTOM_OP(dynamic_bidirectional_rnn, 7, 4, false, 0, 0);

}
}
#endif