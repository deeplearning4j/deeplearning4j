//
//  @author raver119@gmail.com
//

#include <NDArray.h>
#include <helpers/helper_random.h>

namespace nd4j {
    template <typename T>
    class RandomLauncher {
    public:
        static void applyDropOut(nd4j::random::RandomBuffer* buffer, NDArray<T> *array, T retainProb, NDArray<T>* z = nullptr);
        static void applyInvertedDropOut(nd4j::random::RandomBuffer* buffer, NDArray<T> *array, T retainProb, NDArray<T>* z = nullptr);
        static void applyAlphaDropOut(nd4j::random::RandomBuffer* buffer, NDArray<T> *array, T retainProb, T alpha, T beta, T alphaPrime, NDArray<T>* z = nullptr);

        static void fillUniform(nd4j::random::RandomBuffer* buffer, NDArray<T>* array, T from, T to);
        static void fillGaussian(nd4j::random::RandomBuffer* buffer, NDArray<T>* array, T mean, T stdev);
        static void fillLogNormal(nd4j::random::RandomBuffer* buffer, NDArray<T>* array, T mean, T stdev);
        static void fillTruncatedNormal(nd4j::random::RandomBuffer* buffer, NDArray<T>* array, T mean, T stdev);
        static void fillBinomial(nd4j::random::RandomBuffer* buffer, NDArray<T>* array, int trials, T prob);
        static void fillBernoulli(nd4j::random::RandomBuffer* buffer, NDArray<T>* array, T prob);
    };
}