//
//  @author sgazeos@gmail.com
//

#include <ResultSet.h>
#include <ops/declarable/helpers/reduce_dot.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void reduceDotBP(NDArray<T>* inputX, NDArray<T>* inputY, NDArray<T>* epsilon, NDArray<T>* output, std::vector<int> const& axes) {
//                std::unique_ptr<ResultSet<T>> outList(output->allTensorsAlongDimension(dimensions));
                std::vector<int> dimensions; //(input->rankOf() - axes.size());
                for (Nd4jLong e = 0; e < inputX->rankOf(); e++) {
                    if (std::find(axes.begin(), axes.end(), e) == axes.end()) {
                        dimensions.emplace_back(e);
                    }
                }
                std::unique_ptr<ResultSet<T>> outList(output->allTensorsAlongDimension(dimensions));
                std::unique_ptr<ResultSet<T>> yList(inputY->allTensorsAlongDimension(dimensions));
                //output->
#pragma omp parallel for if (outList->size() > Environment::getInstance()->elementwiseThreshold()) schedule(static) 
                for (Nd4jLong e = 0; e < outList->size(); ++e) {
                    outList->at(e)->assign(epsilon);
                    outList->at(e)->template applyPairwiseTransform<simdOps::Multiply<T>>(yList->at(e), outList->at(e), nullptr);
                }

    }

    template void reduceDotBP(NDArray<float>* inputX,  NDArray<float>* inputY,   NDArray<float>* epsilon, NDArray<float>* output, std::vector<int> const& axes);
    template void reduceDotBP(NDArray<float16>* inputX,NDArray<float16>* inputY, NDArray<float16>* epsilon, NDArray<float16>* output, std::vector<int> const& axes);
    template void reduceDotBP(NDArray<double>* inputX, NDArray<double>* inputY,  NDArray<double>* epsilon, NDArray<double>* output, std::vector<int> const& axes);
}
}
}