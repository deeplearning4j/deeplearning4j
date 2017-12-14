//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        
    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operations for Simple Recurrent Unit: "Training RNNs as Fast as CNNs" Tao Lei, Yu Zhang, Yoav Artzi
       * 
       * Input arrays: 
       *    0: input 3d tensor with shape [bS x K x N], N - number of time steps, bS - batch size, K - number of features
       *    1: 2d tensor of weights [3K x K]
       *    2: row of biases with twice length [1 × 2K]
       *    3: 2d tensor of previous cell state [bS x K]
       *    4: optional, 2d tensor of dropout mask [bS x K]
       *  
       * Output arrays: 
       *    0: 3d tensor of cell output [bS x K x N]
       *    1: 3d tensor of cell state [bS x K x N]
       */                  
        DECLARE_CUSTOM_OP(sru,         5, 2, false, 0, 0);

    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operation for Simple Recurrent Unit: "Training RNNs as Fast as CNNs" Tao Lei, Yu Zhang, Yoav Artzi
       * 
       * Input arrays: 
       *    0: input 3d tensor with shape [bS x K x N], N - number of time steps, bS - batch size, K - number of features
       *    1: 2d tensor of weights [3K x K]
       *    2: row of biases with twice length [1 × 2K]
       *    3: 2d tensor of previous cell state [bS x K]
       *    4: optional, 2d tensor of dropout mask [bS x K]
       *  
       * Output arrays: 
       *    0: 3d tensor of cell output [bS x K x N]
       *    1: 3d tensor of cell state [bS x K x N]
       */                  
        DECLARE_CUSTOM_OP(sru_logic,   5, 2, false, 0, 0);


    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operation for Simple Recurrent Unit (bidirectional case): "Training RNNs as Fast as CNNs" Tao Lei, Yu Zhang, Yoav Artzi
       * 
       * Input arrays: 
       *    0: input 3d tensor with shape [N x bS x 2K], N - number of time steps, bS - batch size, K - number of features
       *    1: 2d tensor of weights [2K x 6K]
       *    2: row of biases with twice length [1 × 4K]
       *    3: 2d tensor of previous cell state [bS x 2K]
       *    4: optional, 2d tensor of dropout mask [bS x 2K]
       *  
       * Output arrays: 
       *    0: 3d tensor of cell output [N x bS x 2K]
       *    1: 3d tensor of cell state [N x bS x 2K]
       */                  
        DECLARE_CUSTOM_OP(sru_bi,      5, 2, true,  0, 0);


    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operation for back propagation in Simple Recurrent Unit: "Training RNNs as Fast as CNNs" Tao Lei, Yu Zhang, Yoav Artzi
       * 
       * Input arrays: 
       *    0: input 3d tensor with shape [bS x K x N], N - number of time steps, bS - batch size, K - number of features
       *    1: 2d tensor of weights [3K x K]
       *    2: row of biases with twice length [1 × 2K]
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
        DECLARE_CUSTOM_OP(sru_bp,      8, 4, true,  0, 0);

    
    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operation for back propagation in Simple Recurrent Unit: "Training RNNs as Fast as CNNs" Tao Lei, Yu Zhang, Yoav Artzi
       * 
       * Input arrays: 
       *    0: input 3d tensor with shape [bS x K x N], N - number of time steps, bS - batch size, K - number of features
       *    1: 2d tensor of weights [3K x K]
       *    2: row of biases with twice length [1 × 2K]
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
        DECLARE_CUSTOM_OP(sru_bp_logic,8, 4, true,  0, 0);


    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operation for back propagation in Simple Recurrent Unit (bidirectional case): "Training RNNs as Fast as CNNs" Tao Lei, Yu Zhang, Yoav Artzi
       * 
       * Input arrays: 
       *    0: input 3d tensor with shape [N x bS x 2K], N - number of time steps, bS - batch size, K - number of features
       *    1: 2d tensor of weights [2K x 6K]
       *    2: row of biases with twice length [1 × 4K]
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
        DECLARE_CUSTOM_OP(sru_bi_bp,   8, 4, true,  0, 0);


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
       *    5: diagonal weights for peephole connections [1 x 3*numUnits] 
       *    6: projection weights [numUnits x numProj] 
       *    7: biases, [1 x 4*numUnits] 
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
        DECLARE_CUSTOM_OP(lstmCell, 8, 2, false, 3, 2);

    
    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of operations for Simple Recurrent Unit cell: "Training RNNs as Fast as CNNs" Tao Lei, Yu Zhang, Yoav Artzi
       *
       * Input arrays: 
       *    0: input with shape [batchSize x inSize], batchSize - batch size, inSize - number of features
       *    1: previous cell state [batchSize x inSize], that is at previous time step t-1
       *    2: weights [inSize x 3*inSize]
       *    3: biases [1 × 2*inSize]
       * 
       * Output arrays: 
       *    0: current cell output [batchSize x inSize], that is at current time step t
       *    1: current cell state  [batchSize x inSize], that is at current time step t
       */                  
        DECLARE_CUSTOM_OP(sruCell, 4, 2, false, 0, 0);


    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of gated Recurrent Unit cell:
       *    Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio       
       *    "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
       *
       * Input arrays: 
       *    0: input with shape [batchSize x inSize], batchSize - batch size, inSize - number of features
       *    1: previous cell output [batchSize x numUnits],  that is at previous time step t-1
       *    2: input-to-hidden  weights, [inSize   x 3*numUnits] 
       *    3: hidden-to-hidden weights, [numUnits x 3*numUnits] 
       *    4: biases, [1 x 3*numUnits]        
       *  
       * Output arrays: 
       *    0: current cell output [batchSize x numUnits], that is at current time step t       
       */                  
        DECLARE_CUSTOM_OP(gruCell, 5, 1, false, 0, 0);
    }
}