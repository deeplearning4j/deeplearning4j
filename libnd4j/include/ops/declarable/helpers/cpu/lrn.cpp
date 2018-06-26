//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/lrn.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int lrnFunctor(NDArray<T>* input, NDArray<T>* output, int depth, T bias, T alpha, T beta) {

        T dividor;

        int totalLength = input->lengthOf();
        int lastDim = input->sizeAt(-1);
        int chunkCount = totalLength / lastDim;

        for (int c = 0; c < chunkCount; c++) {
            for (int e = 0; e < lastDim; e++) {
                int begin = nd4j::math::nd4j_max(0, e - depth);
                int end = nd4j::math::nd4j_min(depth + e + 1, lastDim);
                T quadSum = 0;

                for (int pos = begin; pos < end; ++pos) {
                    T val = (*input)(c * lastDim + pos);
                    quadSum += val * val;
                }
                T dividor = nd4j::math::nd4j_pow(bias + alpha * quadSum, beta);
                (*output)(c * lastDim + e) = (*input)(c * lastDim + e) / dividor;
            }
        }

        return ND4J_STATUS_OK;
    }

    template <typename T>
    int lrnFunctorEx(NDArray<T>* input, NDArray<T>* output, NDArray<T>* unitScale, NDArray<T>* scale, int depth, T bias, T alpha, T beta) {
    
        depth = nd4j::math::nd4j_min<Nd4jLong>(depth, input->sizeAt(1));

        int halfDepth = (int) ( (T) depth / (T) 2.f);
        halfDepth = nd4j::math::nd4j_max(halfDepth, 0);
        const int channel =  input->sizeAt(1);

        std::unique_ptr<NDArray<T>> activitySqr(input->dup('c'));//NDArrayFactory<T>::createUninitialized(input));
        std::unique_ptr<NDArray<T>> sumPart(activitySqr->dup('c'));

        input->template applyPairwiseTransform<simdOps::Multiply<T>>(input, activitySqr.get(), nullptr);
#pragma omp parallel for if (halfDepth + 1 > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
        for (int i = 1; i < halfDepth + 1; i++) {
            IndicesList indA({NDIndex::all(), NDIndex::interval(i, channel), NDIndex::all(), NDIndex::all()});
            IndicesList indB({NDIndex::all(), NDIndex::interval(0, channel - i), NDIndex::all(), NDIndex::all()});

            std::unique_ptr<NDArray<T>> tmp(sumPart->subarray(indA));
            std::unique_ptr<NDArray<T>> addVal(activitySqr->subarray(indB));

            tmp->template applyPairwiseTransform<simdOps::Add<T>>(addVal.get(), nullptr);


            std::unique_ptr<NDArray<T>> tmp2(sumPart->subarray(indB));
            std::unique_ptr<NDArray<T>> addVal2(activitySqr->subarray(indA));

            tmp2->template applyPairwiseTransform<simdOps::Add<T>>(addVal2.get(), nullptr);
        }

        /*
         *  // taken from java
            unitScale = sumPart.mul(alpha).addi(k).leverageTo(ComputationGraph.workspaceExternal);
            // y = x * unitScale**-beta
            scale = Transforms.pow(unitScale, -beta).leverageTo(ComputationGraph.workspaceExternal);
            activations = input.mul(scale).leverageTo(ComputationGraph.workspaceExternal);
         */
        if (unitScale != nullptr && scale != nullptr) {
            sumPart->template applyScalar<simdOps::Multiply<T>>(alpha, unitScale, nullptr);
            unitScale->template applyScalar<simdOps::Add<T>>(bias);

            T p = -beta;
            unitScale->template applyTransform<simdOps::Pow<T>>(scale, &p);
            input->template applyPairwiseTransform<simdOps::Multiply<T>>(scale, output, nullptr);
        }

        return ND4J_STATUS_OK;
    }

    template int lrnFunctor(NDArray<float>* input, NDArray<float>* output, int depth, float bias, float alpha, float beta);
    template int lrnFunctor(NDArray<float16>* input, NDArray<float16>* output, int depth, float16 bias, float16 alpha, float16 beta);
    template int lrnFunctor(NDArray<double>* input, NDArray<double>* output, int depth, double bias, double alpha, double beta);
    template int lrnFunctorEx(NDArray<float>* input, NDArray<float>* output, NDArray<float>* unitScale, NDArray<float>* scale, int depth, float bias, float alpha, float beta);
    template int lrnFunctorEx(NDArray<float16>* input, NDArray<float16>* output, NDArray<float16>* unitScale, NDArray<float16>* scale, int depth, float16 bias, float16 alpha, float16 beta);
    template int lrnFunctorEx(NDArray<double>* input, NDArray<double>* output, NDArray<double>* unitScale, NDArray<double>* scale, int depth, double bias, double alpha, double beta);
}
}
}