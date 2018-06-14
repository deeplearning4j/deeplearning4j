//
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/reduce_product.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void reduceProdBP(NDArray<T>* input, NDArray<T>* epsilon, NDArray<T>* output, std::vector<int> const& axes) {

        std::vector<int> dimensions; //(input->rankOf() - axes.size());

#pragma omp parallel for if input->rankOf >  Environment::getInstance()->elementwiseThreshold()) schedule(static)        for (Nd4jLong e = 0; e < input->rankOf(); e++) {
            if (std::find(axes.begin(), axes.end(), e) == axes.end()) {
                dimensions.emplace_back(e);
            }
        }
        std::unique_ptr<ResultSet<T>> outList(NDArrayFactory<T>::allTensorsAlongDimension(output, dimensions));
        std::unique_ptr<ResultSet<T>> inList(NDArrayFactory<T>::allTensorsAlongDimension(input, dimensions));
#pragma omp parallel for if outList->size > Environment::getInstance()->elementwiseThreshold()) schedule(static) 
        for (Nd4jLong e = 0; e < outList->size(); ++e) {
            outList->at(e)->assign(epsilon);
            outList->at(e)->template applyPairwiseTransform<simdOps::Multiply<T>>(tempProd, nullptr);
            outList->at(e)->template applyPairwiseTransform<simdOps::Divide<T>>(inList->at(e), nullptr);
        }

    }

    template void reduceProdBP(NDArray<float>* input,   NDArray<float>*   epsilon, NDArray<float>*   output, std::vector<int> const& axes);
    template void reduceProdBP(NDArray<float16>* input, NDArray<float16>* epsilon, NDArray<float16>* output, std::vector<int> const& axes);
    template void reduceProdBP(NDArray<double>* input,  NDArray<double>*  epsilon, NDArray<double>*  output, std::vector<int> const& axes);
}
}
}