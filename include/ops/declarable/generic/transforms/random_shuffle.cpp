//
// Created by Yurii Syrma on 26.01.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_random_shuffle)

#include <ops/declarable/CustomOperations.h>
#include <numeric>

namespace nd4j {
namespace ops {

OP_IMPL(random_shuffle, 1, 1, true) {
    
    NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* output = nullptr;
    if(!block.isInplace())
       output = OUTPUT_VARIABLE(0);
    
    REQUIRE_TRUE(block.getRNG() != nullptr, 0, "RANDOM_SHUFFLE op: RNG should be defined in Graph !");

    // check edge cases first
    int temp;
    const int firstDim = input->sizeAt(0);    
    if(input->lengthOf() == 1 || firstDim == 1) {
        
        if(!block.isInplace())
            output->assign(input);
    } 
    else if (input->isVector() || shape::isLikeVector(input->getShapeInfo(), temp)) {
        
        // get instance of random generator    
        nd4j::random::RandomBuffer* rng = block.getRNG();   
        // apply Fisher-Yates shuffle 
        if(block.isInplace()) {
// #pragma omp parallel for schedule(guided)        
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng->nextInt(0, i);
                if(i == r)
                    continue;
                math::nd4j_swap<T>((*input)(i), (*input)(i));            
            }        
        }
        else {        
            std::vector<int> indeces(firstDim);        
            std::iota(indeces.begin(), indeces.end(), 0);        
            (*output)(0) = (*input)(0);
// #pragma omp parallel for schedule(guided)        
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng->nextInt(0, i);
                (*output)(i) = (*input)(indeces[r]);
                if(i == r)
                    continue;
                (*output)(r) = (*input)(indeces[i]);                
                math::nd4j_swap<int>(indeces[i], indeces[r]);
            }           
            rng->rewindH(firstDim-1);
        }
    }
    else {
    
        nd4j::random::RandomBuffer* rng = block.getRNG();    
    
        // evaluate sub-arrays list of input array through all dimensions excluding first one
        std::vector<int> dimensions = ShapeUtils<T>::evalDimsToExclude(input->rankOf(), {0});       
        ResultSet<T>* subArrsListIn = NDArrayFactory<T>::allTensorsAlongDimension(input, dimensions);

        // apply Fisher-Yates shuffle
        if(block.isInplace()) {
// #pragma omp parallel for schedule(guided)        
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng->nextInt(0, i);
                if(i == r)
                    continue;
                subArrsListIn->at(i)->swapUnsafe(*subArrsListIn->at(r));
            }        
        }
        else {
            // evaluate sub-arrays list of output array through all dimensions excluding first one        
            ResultSet<T>* subArrsListOut = NDArrayFactory<T>::allTensorsAlongDimension(output, dimensions);        
            std::vector<int> indeces(firstDim);        
            std::iota(indeces.begin(), indeces.end(), 0);        
            bool isZeroShuffled = false;
// #pragma omp parallel for schedule(guided)        
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng->nextInt(0, i);
                subArrsListOut->at(i)->assign(subArrsListIn->at(indeces[r]));
                if(r == 0)
                    isZeroShuffled = true;
                if(i == r)
                    continue;
                subArrsListOut->at(r)->assign(subArrsListIn->at(indeces[i]));
                math::nd4j_swap<int>(indeces[i], indeces[r]);
            }           
            if(!isZeroShuffled)
                subArrsListOut->at(0)->assign(subArrsListIn->at(0));
            delete subArrsListOut;
        }
        rng->rewindH(firstDim-1);
        delete subArrsListIn;
    }
    
    return Status::OK();
}


}
}

#endif