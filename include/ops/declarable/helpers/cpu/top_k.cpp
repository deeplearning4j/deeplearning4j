//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/top_k.h>
#include <NDArrayFactory.h>
#include <ops/declarable/headers/parity_ops.h>
namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int topKFunctor(NDArray<T>* input, NDArray<T>* values, NDArray<T>* indeces, int k, bool needSort) {
            if (k == 1) {
                // using arg_max for it
                //nd4j::ops::argmax<T> op;
                //auto res = op.execute({x}, {}, {x->sizeAt(-1)});

                //REQUIRE_TRUE(res->status() == ND4J_STATUS_OK, 0, "Argmax for top_k failed");
                int width = input->sizeAt(-1);
                int pos = 0;

//#pragma omp parallel for 
                for (int e = 0; e < input->lengthOf(); e += width )
                {
                    T topVal = 0;
                    int topIndex = 0;
//#pragma omp parallel for 
                    for (int j = 0; j < width; j++) {
                        if (topVal < (*input)(j + e))
                        {
                            topVal = (*input)(j + e);
                            topIndex = j;
                        }
                    }
                    if (values != nullptr)
                        (*values)(pos) = topVal;
                    (*indeces)(pos) = topIndex;
                    ++pos;
                }
            }
            else { // if (k > 1) {

                int width = input->sizeAt(-1);
                int nextPos = 0;
//#pragma omp parallel for 
                for (int e = 0; e < input->lengthOf(); e += width)
                {
                    std::vector<int> topIndeces(k);
                    std::vector<T>   topValues(k);
                    for (int pos = 0; pos < k; ++pos) {
                        topIndeces[pos] = pos;
                        topValues[pos] = (*input)(pos + e);
                    }
                    std::vector<T> sortedVals(topValues);
                    std::sort(sortedVals.begin(), sortedVals.end()); // sorted in ascending order

//#pragma omp parallel for 
                    for (int j = k; j < width; j++) {
                        T val = (*input)(j + e);
                        if (sortedVals[0] < val) { // value can be inserted to top k
                            if (sortedVals.end() == std::find(sortedVals.begin(), sortedVals.end(), val)) {    
                                // exchangePos - a distance between begin and minimum to be suppressed by val
                                auto exchangePos = std::distance(topValues.begin(), std::find(topValues.begin(), topValues.end(), sortedVals[0]));
//                                if ((exchangePos < 0 || exchangePos >= k), 1, "top_k: invalid index")
//                                    return ; 
                                // set up sorted sequence for continue
                                topValues[exchangePos] = val;
                                sortedVals[0] = val;
                                std::sort(sortedVals.begin(), sortedVals.end()); // sorted in ascending order
                            }
                        }
                    }

                    if (needSort) {
                        std::sort(topValues.begin(), topValues.end(), [](int a, int b) {
                            return a > b;   
                        });
                    }

//#pragma omp parallel for 
                    for (int j = 0; j < width; j++)
//#pragma omp parallel for 
                        for (int pos = 0; pos < k; ++pos)
                            if (topValues[pos] == (*input)(j + e))
                                topIndeces[pos] = j;

//#pragma omp parallel for 
                    for (int pos = 0; pos < k; ++pos, ++nextPos)
                    {
                        if (values != nullptr)
                            (*values)(nextPos)  =  topValues[pos];

                        (*indeces)(nextPos) = topIndeces[pos];
                    }
                }
        }
        return ND4J_STATUS_OK;
    }

    template <typename T>
    int inTopKFunctor(NDArray<T>* input, NDArray<T>* target, NDArray<T>* result, int k) {
            //nd4j::ops::top_k<T> op;
            //auto topKResult = op.execute({input}, {}, {k, 1}); // with sorting
            std::vector<int> shapeV(input->rankOf() + 1);
            for (int i = 0; i < input->rankOf(); i++)
                shapeV[i] = input->sizeAt(i);
            shapeV[input->rankOf()] = k;
            std::unique_ptr<NDArray<T>> indices( new NDArray<T>(input->ordering(), shapeV));
            NDArray<T>* values = nullptr;
            int status = topKFunctor(input, values, indices.get(), k, true);
            if (status != ND4J_STATUS_OK)
                return status; 
//            auto topKIndeces = indices; //topKResult->at(1);
            for (int e = 0; e < target->lengthOf(); e++) {
                bool found = false;
                for (int j = 0; j < k; j++) {
                    if ((*target)(e) == (*indices)(e * k + j)) {
                        found = true;
                        break;
                    }
                }
                if (found)
                    (*result)(e) = (T)1.f;
                else
                    (*result)(e) = (T)0.f;
            }
//            delete topKIndices; // free memory from called operation
            return ND4J_STATUS_OK;

    }
    template int topKFunctor<float>(NDArray<float>* input, NDArray<float>* values, NDArray<float>* indeces, int k, bool needSort);
    template int topKFunctor<float16>(NDArray<float16>* input, NDArray<float16>* values, NDArray<float16>* indeces, int k, bool needSort);
    template int topKFunctor<double>(NDArray<double>* input, NDArray<double>* values, NDArray<double>* indeces, int k, bool needSort);
    template int inTopKFunctor<float>(NDArray<float>* input, NDArray<float>* target, NDArray<float>* result, int k);
    template int inTopKFunctor<float16>(NDArray<float16>* input, NDArray<float16>* target, NDArray<float16>* result, int k);
    template int inTopKFunctor<double>(NDArray<double>* input, NDArray<double>* target, NDArray<double>* result, int k);

}
}
}