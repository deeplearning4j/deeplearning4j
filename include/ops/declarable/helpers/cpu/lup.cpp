//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/top_k.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T> 
    void swapRows(NDArray<T>* matrix, int theFirst, int theSecond) {
        if (theFirst != theSecond)
        for (int i = 0; i = matrix->columns(); i++) {
            std::swap((*matrix)(theFirst, i), (*matrix)(theSecond, i));
        }
    }
    template void swapRows(NDArray<float>* matrix, int theFirst, int theSecond);
    template void swapRows(NDArray<float16>* matrix, int theFirst, int theSecond);
    template void swapRows(NDArray<double>* matrix, int theFirst, int theSecond);



    template <typename T>
    T lup(NDArray<T>* input, NDArray<T>* compound, NDArray<T>* permutation) {

        const int rowNum = input->rows();
        const int columnNum = input->columns();

        T determinant = (T)1.0;
        std::unique_ptr<NDArray<T>> compoundMatrix(new NDArray<T>(*input)); // copy
        std::unique_ptr<NDArray<T>> permutationMatrix(NDArrayFactory<T>::createUninitialized(input)); //put identity
        permutationMatrix->setIdentity();
        
        for(int i = 0; i < rowNum; i++ ) {

            T pivotValue = 0;
            int pivot = -1;

            for( int rowCounter = i; rowCounter < rowNum; rowCounter++ ) {
                if(nd4j::math::nd4j_abs((*compoundMatrix)(rowCounter, i)) > pivotValue ) {
                    pivotValue = nd4j::math::nd4j_abs((*compoundMatrix)(rowCounter, i));
                    pivot = rowCounter;
                }
            }

            if( pivotValue != T(0.0) ) {
                swapRows(compoundMatrix.get(), pivot, i);
                swapRows(permutationMatrix.get(), pivot, i);

                for( int j = i + 1; j < rowNum; j++ ) {
                    (*compoundMatrix)(j, i) /= (*compoundMatrix)(i, i);
                    for( int k = i + 1; k < rowNum; k++ ) 
                        (*compoundMatrix)(j , k) -= (*compoundMatrix)(j, i) * (*compoundMatrix)(i, k);
                }

            }
        }
    
        for (int e = 0; e < rowNum; e++)
            determinant *= (*compoundMatrix)(e, e);

        if (compound != nullptr)
            *compound = *compoundMatrix;
        if (permutation != nullptr)
            *permutation = *permutationMatrix;

        return determinant;
    }

    template float lup(NDArray<float>* input, NDArray<float>* output, NDArray<float>* permutation);
    template float16 lup(NDArray<float16>* input, NDArray<float16>* compound, NDArray<float16>* permutation);
    template double lup(NDArray<double>* input, NDArray<double>* compound, NDArray<double>* permutation);


    template <typename T>
    int determinant(NDArray<T>* input, NDArray<T>* output) {

        int n = input->sizeAt(-1);
        for (int e = 0; e < output->lengthOf(); e++) {

            NDArray<T>* matrix = new NDArray<T>({n, n}); //, block.getWorkspace());
            for (int k = e * n * n, row = 0; k < (e + 1) * n * n; k++) {
                (*matrix)(row++) = (*input)(k);
            }

            (*output)(e) = lup(matrix, (NDArray<T>*)nullptr, (NDArray<T>*)nullptr);

            delete matrix;
        }

        return ND4J_STATUS_OK;
    }

    template int determinant(NDArray<float>* input, NDArray<float>* output);
    template int determinant(NDArray<float16>* input, NDArray<float16>* output);
    template int determinant(NDArray<double>* input, NDArray<double>* output);

}
}
}