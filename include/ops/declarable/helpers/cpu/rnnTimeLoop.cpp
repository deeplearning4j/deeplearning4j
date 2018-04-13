//
// @author Yurii Shyrma, created on 04.03.2017
//

#include<ops/declarable/helpers/rnnTimeLoop.h>
#include<ops/declarable/helpers/rnnCell.h>
#include <helpers/BlasHelper.h>

namespace nd4j    {
namespace ops     {
namespace helpers {




//////////////////////////////////////////////////////////////////////////
template <typename T>
void rnnTimeLoop(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* h, NDArray<T>* hFinal) {

    NDArray<T>* x  = inArrs[0];               	// input [time x bS x inSize]
	NDArray<T>* Wx = inArrs[1];               	// input-to-hidden  weights, [inSize  x numUnits] 	
    NDArray<T>* Wh = inArrs[2];               	// hidden-to-hidden weights, [numUnits x numUnits]         
	NDArray<T>* b  = inArrs[3];               	// biases for, [2*numUnits] 

	NDArray<T>* h0          = inArrs[4];		// initial cell output (at time step = 0) [bS x numUnits]	
	NDArray<T>* maxTimeStep = inArrs[5];     	// vector [bS] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep
    
    const int time     = x->sizeAt(0);
    const int bS       = x->sizeAt(1);        
    
    // at first time step
    if(h0)
        hFinal->assign(h0);
    else 
        *hFinal = 0.;   

    BlasHelper::getInstance();          // to avoid memory leak in pragma parallel loops
#pragma omp parallel for if(bS > Environment::getInstance()->elementwiseThreshold()) schedule(guided) 
    // loop through batch of inputs           
    for (int e = 0; e < bS; ++e) {              
        
        int maxStep = maxTimeStep ? (int)(*maxTimeStep)(e) : time;
        
        // loop through time steps
        for (int t = 0; t < time; ++t) {                                 

            NDArray<T> xt   = (*x)({{t,t+1}, {e,e+1}, {}}, true);
            NDArray<T> ht   = (*h)({{t,t+1}, {e,e+1}, {}}, true);
            NDArray<T> ht_1 = (*hFinal)({{e,e+1}, {}}, true);                       // previous state 
            
            if(t >= maxStep) {
                ht = 0.;
                if(maxStep != 0)                    
                    ht_1.assign((*h)({{maxStep-1,maxStep}, {e,e+1}, {}}));
            }
            else {
                helpers::rnnCell<T>({&xt, Wx, Wh, b, &ht_1}, &ht);
                ht_1.assign(ht);
            }
        }
    }    
}



template void rnnTimeLoop<float>  (const std::vector<NDArray<float>*>&   inArrs, NDArray<float>*   h, NDArray<float>*   hFinal);
template void rnnTimeLoop<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>* h, NDArray<float16>* hFinal);
template void rnnTimeLoop<double> (const std::vector<NDArray<double>*>&  inArrs, NDArray<double>*  h, NDArray<double>*  hFinal);


}
}
}

