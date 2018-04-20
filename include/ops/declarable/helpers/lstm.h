//
// @author Yurii Shyrma, created on 14.02.2018
//

#ifndef LIBND4J_LSTM_H
#define LIBND4J_LSTM_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


	template <typename T>
	void lstmCell(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs, const std::vector<T>& params);

	template <typename T>
	void lstmTimeLoop(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs, const std::vector<T>& params);
	
    

}
}
}


#endif //LIBND4J_LSTM_H
