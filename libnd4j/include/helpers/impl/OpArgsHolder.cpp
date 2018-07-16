//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 15.07.2018
//

#include <OpArgsHolder.h>


namespace nd4j {

////////////////////////////////////////////////////////////////////////
template <typename T>
OpArgsHolder<T> OpArgsHolder<T>::createArgsHolderForBP(const std::vector<NDArray<T>*>& inGradArrs, const bool isInPlace) const {
	
	const int numInGradArrs = inGradArrs.size();

	OpArgsHolder<T> result(std::vector<NDArray<T>*>(_numInArrs + numInGradArrs, nullptr), _tArgs, _iArgs);

	for (int i = 0; i < _numInArrs; ++i) {
		
		if(isInPlace) {
			result._inArrs[i] = new NDArray<T>(*_inArrs[i]);		// make copy
			result._isArrAlloc[i] = true;
		}
		else
			result._inArrs[i] = _inArrs[i];	
	}

	// input gradients 
	for (int i = 0; i < numInGradArrs; ++i)
		result._inArrs[_numInArrs + i] = inGradArrs[i];

	return result;
}

////////////////////////////////////////////////////////////////////////
// default destructor
template <typename T>
OpArgsHolder<T>::~OpArgsHolder() noexcept {
	
	for (int i = 0; i < _isArrAlloc.size(); ++i)
		if(_isArrAlloc[i])
			delete _inArrs[i];
        
}

template class ND4J_EXPORT OpArgsHolder<float>;
template class ND4J_EXPORT OpArgsHolder<float16>;
template class ND4J_EXPORT OpArgsHolder<double>;

}


