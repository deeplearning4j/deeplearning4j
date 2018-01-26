//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/top_k.h>
#include <NDArrayFactory.h>

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
                for (int e = 0; e < input->lengthOf(); e += width)
                {
                    T topVal = 0;
                    int topIndex = 0;
                    for (int j = 0; j < width; j++) {
                        if (topVal < input->getScalar(j + e))
                        {
                            topVal = input->getScalar(j + e);
                            topIndex = j;
                        }
                    }
                    values->putScalar(pos, topVal);
                    indeces->putScalar(pos++, topIndex);
                }
                //int index = indeces->getScalar(0);
                //T val = x->getScalar(index);
                
                //values->putScalar(0, val);

                //return ND4J_STATUS_OK;
            }
            else { // if (k > 1) {

                int width = input->sizeAt(-1);
                int nextPos = 0;
                for (int e = 0; e < input->lengthOf(); e += width)
                {
                    std::vector<int> topIndeces(k);
                    std::vector<T>   topValues(k);
                    for (int pos = 0; pos < k; ++pos) {
                        topIndeces[pos] = pos;
                        topValues[pos] = input->getScalar(pos + e);
                    }
                    std::vector<T> sortedVals(topValues);
                    std::sort(sortedVals.begin(), sortedVals.end()); // sorted in ascending order
                    
                    for (int j = k; j < width; j++) {
                        if (sortedVals[0] < input->getScalar(j + e)) { // value can be inserted to top k
                            T val = input->getScalar(j + e);
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

                    for (int j = 0; j < width; j++)
                        for (int pos = 0; pos < k; ++pos)
                            if (topValues[pos] == input->getScalar(j + e))
                                topIndeces[pos] = j;

                    for (int pos = 0; pos < k; ++pos)
                    {
                        values->putScalar(nextPos, topValues[pos]);
                        indeces->putScalar(nextPos++, topIndeces[pos]);
                    }
                }
        }
        return ND4J_STATUS_OK;
    }
    template int topKFunctor<float>(NDArray<float>* input, NDArray<float>* values, NDArray<float>* indeces, int k, bool needSort);
    template int topKFunctor<float16>(NDArray<float16>* input, NDArray<float16>* values, NDArray<float16>* indeces, int k, bool needSort);
    template int topKFunctor<double>(NDArray<double>* input, NDArray<double>* values, NDArray<double>* indeces, int k, bool needSort);

}
}
}