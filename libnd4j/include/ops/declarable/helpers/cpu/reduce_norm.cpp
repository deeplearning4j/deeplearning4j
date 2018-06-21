//
//  @author sgazeos@gmail.com
//

#include <ResultSet.h>
#include <NDArrayFactory.h>
#include <ops/declarable/helpers/reduce_product.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void reduceNorm1BP(NDArray<T>* input, NDArray<T>* epsilon, NDArray<T>* tempNorm, NDArray<T>* output, std::vector<int> const& axes) {

        std::vector<int> dimensions; //(input->rankOf() - axes.size());
        for (Nd4jLong e = 0; e < input->rankOf(); e++) {
            if (std::find(axes.begin(), axes.end(), e) == axes.end()) {
                dimensions.emplace_back(e);
            }
        }
        std::unique_ptr<ResultSet<T>> outList(NDArrayFactory<T>::allTensorsAlongDimension(output, dimensions));
        std::unique_ptr<ResultSet<T>> inList(NDArrayFactory<T>::allTensorsAlongDimension(input, dimensions));
        for (int e = 0; e < outList->size(); ++e) {
            auto norm1Backprop = LAMBDA_TT(_x, _e) {
                return (_x >= T(0.f) ?_e:-_e);
            };
            inList->at(e)->applyPairwiseLambda(epsilon, norm1Backprop, outList->at(e));
        }
    }

    template <typename T>
    void reduceNorm2BP(NDArray<T>* input, NDArray<T>* epsilon, NDArray<T>* tempNorm, NDArray<T>* output, std::vector<int> const& axes) {

        std::vector<int> dimensions; //(input->rankOf() - axes.size());
        for (Nd4jLong e = 0; e < input->rankOf(); e++) {
            if (std::find(axes.begin(), axes.end(), e) == axes.end()) {
                dimensions.emplace_back(e);
            }
        }
        std::unique_ptr<ResultSet<T>> outList(NDArrayFactory<T>::allTensorsAlongDimension(output, dimensions));
        std::unique_ptr<ResultSet<T>> inList(NDArrayFactory<T>::allTensorsAlongDimension(input, dimensions));
        for (int e = 0; e < outList->size(); ++e) {
            epsilon->template applyPairwiseTransform<simdOps::Multiply<T>>(inList->at(e), outList->at(e), nullptr);
            outList->at(e)->template applyPairwiseTransform<simdOps::Divide<T>>(tempNorm, outList->at(e), nullptr);
        }
    }

    template <typename T>
    void reduceSquareNormBP(NDArray<T>* input, NDArray<T>* epsilon, NDArray<T>* tempNorm, NDArray<T>* output, std::vector<int> const& axes) {

        std::vector<int> dimensions; //(input->rankOf() - axes.size());
        for (Nd4jLong e = 0; e < input->rankOf(); e++) {
            if (std::find(axes.begin(), axes.end(), e) == axes.end()) {
                dimensions.emplace_back(e);
            }
        }
        std::unique_ptr<ResultSet<T>> outList(NDArrayFactory<T>::allTensorsAlongDimension(output, dimensions));
        std::unique_ptr<ResultSet<T>> inList(NDArrayFactory<T>::allTensorsAlongDimension(input, dimensions));
        for (int e = 0; e < outList->size(); ++e) {
            outList->at(e)->assign(T(2.f));
            outList->at(e)->template applyPairwiseTransform<simdOps::Multiply<T>>(epsilon, outList->at(e), nullptr);
            outList->at(e)->template applyPairwiseTransform<simdOps::Multiply<T>>(inList->at(e), outList->at(e), nullptr);
        }
    }

    template void reduceNorm1BP(NDArray<float>* input, NDArray<float>* epsilon, NDArray<float>* tempNorm, NDArray<float>* output, std::vector<int> const& axes);
    template void reduceNorm1BP(NDArray<float16>* input, NDArray<float16>* epsilon, NDArray<float16>* tempNorm, NDArray<float16>* output, std::vector<int> const& axes);
    template void reduceNorm1BP(NDArray<double>* input, NDArray<double>* epsilon, NDArray<double>* tempNorm, NDArray<double>* output, std::vector<int> const& axes);

    template void reduceNorm2BP(NDArray<float>* input, NDArray<float>* epsilon, NDArray<float>* tempNorm, NDArray<float>* output, std::vector<int> const& axes);
    template void reduceNorm2BP(NDArray<float16>* input, NDArray<float16>* epsilon, NDArray<float16>* tempNorm, NDArray<float16>* output, std::vector<int> const& axes);
    template void reduceNorm2BP(NDArray<double>* input, NDArray<double>* epsilon, NDArray<double>* tempNorm, NDArray<double>* output, std::vector<int> const& axes);

    template void reduceSquareNormBP(NDArray<float>* input, NDArray<float>* epsilon, NDArray<float>* tempNorm, NDArray<float>* output, std::vector<int> const& axes);
    template void reduceSquareNormBP(NDArray<float16>* input, NDArray<float16>* epsilon, NDArray<float16>* tempNorm, NDArray<float16>* output, std::vector<int> const& axes);
    template void reduceSquareNormBP(NDArray<double>* input, NDArray<double>* epsilon, NDArray<double>* tempNorm, NDArray<double>* output, std::vector<int> const& axes);

}
}
}