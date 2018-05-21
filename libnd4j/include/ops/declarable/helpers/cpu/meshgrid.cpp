//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 18.04.2018
//


#include<ops/declarable/helpers/meshgrid.h>
#include <array/ResultSet.h>
#include <NDArrayFactory.h>
#include <numeric>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
void meshgrid(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs, const bool swapFirst2Dims) {

    const int rank = inArrs.size();
    int* inIndices = new int[rank];
    std::iota(inIndices, inIndices + rank, 0);
    if(swapFirst2Dims && rank > 1) {
        inIndices[0] = 1;
        inIndices[1] = 0;
    }
            
    for(int i = 0; i < rank; ++i) {        
        ResultSet<T>* list = NDArrayFactory<T>::allTensorsAlongDimension(outArrs[i], {inIndices[i]});        
        for(int j = 0; j < list->size(); ++j)
            list->at(j)->assign(inArrs[i]);

        delete list;
    }    

    delete []inIndices;
    
}


template void meshgrid<float>(const std::vector<NDArray<float>*>& inArrs, const std::vector<NDArray<float>*>& outArrs, const bool swapFirst2Dims);
template void meshgrid<float16>(const std::vector<NDArray<float16>*>& inArrs, const std::vector<NDArray<float16>*>& outArrs, const bool swapFirst2Dims);
template void meshgrid<double>(const std::vector<NDArray<double>*>& inArrs, const std::vector<NDArray<double>*>& outArrs, const bool swapFirst2Dims);

}
}
}

