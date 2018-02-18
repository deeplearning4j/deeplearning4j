//
// Created by Yurii Shyrma on 15.02.2018
//

// implementation of gated Recurrent Unit cell 
// (cf. http://arxiv.org/abs/1406.1078).
// Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio
// "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"


#include<ops/declarable/helpers/gruCell.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
static FORCEINLINE NDArray<T> sigmoid(const NDArray<T>& arr) {    
    
    return (const_cast<NDArray<T>&>(arr)).template transform<simdOps::Sigmoid<T>>();    
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static FORCEINLINE NDArray<T> activation(const NDArray<T>& arr) {    
    
    return (const_cast<NDArray<T>&>(arr)).template transform<simdOps::Tanh<T>>();    
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void gruCell(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* ht) {

    NDArray<T>* xt   = inArrs[0];                   // input [batchSize x inSize]
    NDArray<T>* ht_1 = inArrs[1];                   // previous cell output [batchSize x numUnits],  that is at previous time step t-1

    NDArray<T>* Wx   = inArrs[2];                   // input-to-hidden  weights, [inSize   x 3*numUnits] 
    NDArray<T>* Wh   = inArrs[3];                   // hidden-to-hidden weights, [numUnits x 3*numUnits]     
    NDArray<T>* b    = inArrs[4];                   // biases, [3*numUnits] 
    
    // ht is current cell output [batchSize x numUnits], that is at current time step t    

    const int batchSize   = xt->sizeAt(0);
    const int inSize      = xt->sizeAt(1);
    const int numUnits    = ht_1->sizeAt(1);
    
    // activ = sigmoid(xt*Wx + ht_1*Wh + b)
    NDArray<T> activ = sigmoid<T>(mmul(*xt, (*Wx)({{},{0,2*numUnits}})) + mmul(*ht_1, (*Wh)({{},{0,2*numUnits}})) + (*b)({{0,2*numUnits}}));       // [batchSize x 2*numUnits] + [batchSize x 2*numUnits] + [1 x 2*numUnits] = [batchSize x 2*numUnits]    
    
    // reset gate
    NDArray<T> rt = activ({{}, {0, numUnits}});                     // [batchSize x numUnits]

    // update gate
    NDArray<T> ut = activ({{}, {numUnits, 2*numUnits}});            // [batchSize x numUnits]

    // ht_tilde = activation(xt*Wx + (rt(*)ht_1)*Wh + b)
    NDArray<T> ht_tilde = activation<T>(mmul(*xt, (*Wx)({{},{2*numUnits, 3*numUnits}})) + mmul((*ht_1)*rt, (*Wh)({{},{2*numUnits,3*numUnits}})) + (*b)({{2*numUnits,3*numUnits}}));     // [batchSize x numUnits]

    // current cell output
    *ht = ut * (*ht_1) + ((T)1. - ut) * ht_tilde;
}



template void gruCell<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>* ht);
template void gruCell<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>* ht);
template void gruCell<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>* ht);


}
}
}

