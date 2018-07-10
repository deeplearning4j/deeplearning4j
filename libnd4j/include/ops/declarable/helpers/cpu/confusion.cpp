//
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/helpers/confusion.h>


namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void confusionFunctor(NDArray<T>* labels, NDArray<T>* predictions, NDArray<T>* weights, NDArray<T>* output) {
        std::unique_ptr<ResultSet<T>> arrs(output->allTensorsAlongDimension({1}));

#pragma omp parallel for if(labels->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)                    
        for (int j = 0; j < labels->lengthOf(); ++j){
            Nd4jLong label = (*labels)(j);
            Nd4jLong pred = (*predictions)(j);
            T value = (weights == nullptr ? (T)1.0 : (*weights)(j));
            (*arrs->at(label))(pred) = value;
        }
    }

    template void confusionFunctor(NDArray<float>* labels, NDArray<float>* predictions, NDArray<float>* weights, NDArray<float>* output);
    template void confusionFunctor(NDArray<float16>* labels, NDArray<float16>* predictions, NDArray<float16>* weights, NDArray<float16>* output);
    template void confusionFunctor(NDArray<double>* labels, NDArray<double>* predictions, NDArray<double>* weights, NDArray<double>* output);
}
}
}