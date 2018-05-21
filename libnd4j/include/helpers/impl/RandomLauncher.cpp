//
//  @author raver119@gmail.com
//

#include <types/float16.h>
#include <dll.h>
#include <helpers/RandomLauncher.h>

namespace nd4j {
    template <typename T>
    void RandomLauncher<T>::applyDropOut(nd4j::random::RandomBuffer* buffer, NDArray<T> *array, T retainProb, NDArray<T>* z) {
        if (z == nullptr)
            z = array;

        array->template applyRandom<randomOps::DropOut<T>>(buffer, nullptr, z, &retainProb);
    }

    template <typename T>
    void RandomLauncher<T>::applyInvertedDropOut(nd4j::random::RandomBuffer* buffer, NDArray<T> *array, T retainProb, NDArray<T>* z) {
        if (z == nullptr)
            z = array;

        array->template applyRandom<randomOps::DropOutInverted<T>>(buffer, nullptr, z, &retainProb);
    }

    template <typename T>
    void RandomLauncher<T>::applyAlphaDropOut(nd4j::random::RandomBuffer* buffer, NDArray<T> *array, T retainProb, T alpha, T beta, T alphaPrime, NDArray<T>* z) {
        if (z == nullptr)
            z = array;

        //  FIXME: this isn't portable code :(
        T args[] = {retainProb, alpha, beta, alphaPrime};

        array->template applyRandom<randomOps::AlphaDropOut<T>>(buffer, nullptr, z, args);
    }

    template <typename T>
    void RandomLauncher<T>::fillBernoulli(nd4j::random::RandomBuffer* buffer, NDArray<T>* array, T prob) {
        array->template applyRandom<randomOps::BernoulliDistribution<T>>(buffer, nullptr, array, &prob);
    }

    template <typename T>
    void RandomLauncher<T>::fillUniform(nd4j::random::RandomBuffer* buffer, NDArray<T>* array, T from, T to) {
        T args[] = {from, to};

        array->template applyRandom<randomOps::UniformDistribution<T>>(buffer, nullptr, nullptr, args);
    }

    template <typename T>
    void RandomLauncher<T>::fillGaussian(nd4j::random::RandomBuffer* buffer, NDArray<T>* array, T mean, T stdev) {
        T args[] = {mean, stdev};

        array->template applyRandom<randomOps::GaussianDistribution<T>>(buffer, array, array, args);
    }

    template <typename T>
    void RandomLauncher<T>::fillLogNormal(nd4j::random::RandomBuffer* buffer, NDArray<T>* array, T mean, T stdev) {
        T args[] = {mean, stdev};

        array->template applyRandom<randomOps::LogNormalDistribution<T>>(buffer, array, array, args);
    }

    template <typename T>
    void RandomLauncher<T>::fillTruncatedNormal(nd4j::random::RandomBuffer* buffer, NDArray<T>* array, T mean, T stdev) {
        T args[] = {mean, stdev};

        array->template applyRandom<randomOps::TruncatedNormalDistribution<T>>(buffer, array, array, args);
    }

    template <typename T>
    void RandomLauncher<T>::fillBinomial(nd4j::random::RandomBuffer* buffer, NDArray<T>* array, int trials, T prob) {
        T args[] = {(T) trials, prob};

        array->template applyRandom<randomOps::BinomialDistributionEx<T>>(buffer, array, array, args);
    }

    template class ND4J_EXPORT RandomLauncher<float>;
    template class ND4J_EXPORT RandomLauncher<float16>;
    template class ND4J_EXPORT RandomLauncher<double>;
}