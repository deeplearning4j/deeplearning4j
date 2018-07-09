//
// Created by Yurii Shyrma on 02.01.2018
//

#include <ops/declarable/helpers/stack.h>
#include <helpers/ShapeUtils.h>
#include <array/ResultSet.h>


namespace nd4j {
namespace ops {
namespace helpers {


///////////////////////////////////////////////////////////////////
template <typename T>
void stack(const std::vector<NDArray<T>*>& inArrs, NDArray<T>& outArr, const int dim) {

	if(inArrs[0]->rankOf() == 0) {

#pragma omp parallel for if(inArrs.size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
		for(int i=0; i < inArrs.size(); ++i)
			outArr(i) = (*inArrs[i])(0);
	}
	else {

		std::vector<int> dimsToExclude = ShapeUtils<T>::evalDimsToExclude(outArr.rankOf(), {dim});	
		ResultSet<T>* list = outArr.allTensorsAlongDimension(dimsToExclude);		// list.size() == block.width()
		
#pragma omp parallel for if(list->size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
		for(int i=0; i<list->size(); ++i)
			list->at(i)->assign(inArrs[i]);
		
		delete list;
	}
}


template void stack<float>  (const std::vector<NDArray<float  >*>& inArrs, NDArray<float  >& outArr, const int dim);
template void stack<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>& outArr, const int dim);
template void stack<double> (const std::vector<NDArray<double >*>& inArrs, NDArray<double >& outArr, const int dim);


}
}
}

