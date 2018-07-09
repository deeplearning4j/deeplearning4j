//
//  @author raver119@gmail.com
//

#include <vector>
#include <NDArray.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename T>
            void _bgemm(std::vector<NDArray<T>*>& vA, std::vector<NDArray<T>*>& vB, std::vector<NDArray<T>*>& vC, NDArray<T>* alphas, NDArray<T>* betas, int transA, int transB, int M, int N, int K, int ldA, int ldB, int ldC);
        }
    }
}