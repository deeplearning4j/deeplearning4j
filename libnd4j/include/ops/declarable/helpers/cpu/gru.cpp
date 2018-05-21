//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 15.02.2018
//

// implementation of gated Recurrent Unit cell 
// (cf. http://arxiv.org/abs/1406.1078).
// Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio
// "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"


#include<ops/declarable/helpers/gru.h>

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
void gruCell(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* h) {

    NDArray<T>* x = inArrs[0];                    // input [bS x inSize]
    NDArray<T>* h0 = inArrs[1];                   // previous cell output [bS x numUnits],  that is at previous time step t-1

    NDArray<T>* Wx = inArrs[2];                   // input-to-hidden  weights, [inSize   x 3*numUnits] 
    NDArray<T>* Wh = inArrs[3];                   // hidden-to-hidden weights, [numUnits x 3*numUnits]     
    NDArray<T>* b  = inArrs[4];                   // biases, [3*numUnits] 
    
    // h is current cell output [bS x numUnits], that is at current time step t    

    const int numUnits = h0->sizeAt(1);
    
    // activ = sigmoid(x*Wx + h0*Wh + b)
    NDArray<T> activ = sigmoid<T>(mmul(*x, (*Wx)({{},{0,2*numUnits}})) + mmul(*h0, (*Wh)({{},{0,2*numUnits}})) + (*b)({{0,2*numUnits}}));       // [bS x 2*numUnits] + [bS x 2*numUnits] + [1 x 2*numUnits] = [bS x 2*numUnits]    
    
    // reset gate
    NDArray<T> rt = activ({{}, {0, numUnits}});                     // [bS x numUnits]

    // update gate
    NDArray<T> ut = activ({{}, {numUnits, 2*numUnits}});            // [bS x numUnits]

    // ◦ means element-wise product or so called Hadamard product
    // ht_tilde = activation(x*Wx + (rt◦h0)*Wh + b)
    NDArray<T> ht_tilde = activation<T>(mmul(*x, (*Wx)({{},{2*numUnits, 3*numUnits}})) + mmul((*h0)*rt, (*Wh)({{},{2*numUnits,3*numUnits}})) + (*b)({{2*numUnits,3*numUnits}}));     // [bS x numUnits]

    // current cell output
    h->assign( ut * (*h0) + ((T)1. - ut) * ht_tilde );
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void gruTimeLoop(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* h) {

    NDArray<T>* x  = inArrs[0];                   // input [time x bS x inSize]
    NDArray<T>* h0 = inArrs[1];                   // initial cell output (at time step = 0) [bS x numUnits]

    NDArray<T>* Wx = inArrs[2];                   // input-to-hidden  weights, [inSize   x 3*numUnits] 
    NDArray<T>* Wh = inArrs[3];                   // hidden-to-hidden weights, [numUnits x 3*numUnits]     
    NDArray<T>* b  = inArrs[4];                   // biases, [3*numUnits] 
    
    // h is cell outputs at each time step [time x bS x numUnits]

    const int time = x->sizeAt(0);    

    NDArray<T> ht_1(*h0);

    // loop through time steps
    for (int t = 0; t < time; ++t) {

        NDArray<T> xt = (*x)({{t,t+1}, {}, {}});
        NDArray<T> ht = (*h)({{t,t+1}, {}, {}});

        helpers::gruCell<T>({&xt, &ht_1, Wx, Wh, b}, &ht);
        ht_1.assign(ht);    
    }
}



template void gruCell<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>* h);
template void gruCell<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>* h);
template void gruCell<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>* h);

template void gruTimeLoop<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>* h);
template void gruTimeLoop<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>* h);
template void gruTimeLoop<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>* h);

}
}
}

