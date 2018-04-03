//
// @author Yurii Shyrma, created on 27.03.2017
//

// function implements an Elman RNN cell: output = activation(Wx*x + bx  +  Wh*ht  + bh)


#include<ops/declarable/helpers/rnnCell.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
static FORCEINLINE NDArray<T> activation(const NDArray<T>& arr) {    
    
    return (const_cast<NDArray<T>&>(arr)).template transform<simdOps::Tanh<T>>();    
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void rnnCell(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* ht) {

    NDArray<T>* xt   = inArrs[0];                   // input [bS x inSize]
    NDArray<T>* ht_1 = inArrs[1];                   // previous cell output [bS x numUnits],  that is at previous time step t-1, in case of projection=false -> numUnits=numUnits!!!     

    NDArray<T>* Wx   = inArrs[2];                   // input-to-hidden weights, [inSize  x numUnits] 
    NDArray<T>* Wh   = inArrs[3];                   // hidden-to-hidden weights, [numUnits x numUnits] 
    NDArray<T>* b    = inArrs[4];                   // biases, [2*numUnits]: {0, numUnits} are input-to-hidden biases and {numUnits, 2*numUnits} are hidden-to-hidden biases    

    const int numUnits  = ht_1->sizeAt(1);
    
    // ht is current cell output [bS x numUnits], that is at current time step t        
    ht->assign(activation<T>(mmul(*xt, *Wx) + (*b)({{0, numUnits}})  +  mmul(*ht_1, *Wh) + (*b)({{numUnits, 2*numUnits}})));      // [bS x numUnits] + [numUnits]  +  [bS x numUnits] + [numUnits] = [bS x numUnits]    

}



template void rnnCell<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>* ht);
template void rnnCell<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>* ht);
template void rnnCell<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>* ht);


}
}
}

