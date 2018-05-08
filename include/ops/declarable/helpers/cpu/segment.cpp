//
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/helpers/segment.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    // segment max
    template <typename T>
    void segmentMaxFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = static_cast<int>((*indices)(0));
        if (input->isVector()) {
            T val = (*input)(0);
//#pragma omp parallel for
            for (int e = 1; e < indices->lengthOf(); e++) {
                if (idx == static_cast<int>((*indices)(e))) {
                   // max 
                   val = nd4j::math::nd4j_max(val, (*input)(e));
                }
                else {
                    idx = static_cast<int>((*indices)(e));
                    val = (*input)(e);
                }
                (*output)(idx) = val;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;
            ResultSet<T>* listOfTensors = NDArrayFactory<T>::allTensorsAlongDimension(input, restDims);
            ResultSet<T>* listOfOutTensors = NDArrayFactory<T>::allTensorsAlongDimension(output, restDims);

            int numOfClasses = output->sizeAt(0); // number of classes
            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
            NDArray<T>* maxT = listOfOutTensors->at(idx);

            int pos = 0;
            maxT->assign(listOfTensors->at(0));
            for (int i = 1; i < indices->lengthOf(); i++) {
                if (static_cast<int>((*indices)(i)) == idx) {
#pragma omp parallel for
                    for (int e = 0; e < maxT->lengthOf(); e++) {
                       (*maxT)(e) = nd4j::math::nd4j_max((*maxT)(e), (*listOfTensors->at(i))(e));
                    }
                }
                else {
                    idx = static_cast<int>((*indices)(i));
                    maxT = listOfOutTensors->at(idx);
                    maxT->assign(listOfTensors->at(i));
                }

            }
            delete listOfTensors;
            delete listOfOutTensors;
        }
    }

    // segmen min 
    template <typename T>
    void segmentMinFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = static_cast<int>((*indices)(0));
        if (input->isVector()) {
            T val = (*input)(0);
//#pragma omp parallel for
            for (int e = 1; e < indices->lengthOf(); e++) {
                if (idx == static_cast<int>((*indices)(e))) {
                   // min 
                   val = nd4j::math::nd4j_min(val, (*input)(e));
                }
                else {
                    idx = static_cast<int>((*indices)(e));
                    val = (*input)(e);
                }
                (*output)(idx) = val;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfTensors( NDArrayFactory<T>::allTensorsAlongDimension(input, restDims) );
            std::unique_ptr<ResultSet<T>> listOfOutTensors( NDArrayFactory<T>::allTensorsAlongDimension(output, restDims) );

            int numOfClasses = output->sizeAt(0); // number of classes
            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
            NDArray<T>* minT = listOfOutTensors->at(idx);

            int pos = 0;
            minT->assign(listOfTensors->at(0));
            for (int i = 1; i < indices->lengthOf(); i++) {
                if (static_cast<int>((*indices)(i)) == idx) {
#pragma omp parallel for if(minT->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                    for (int e = 0; e < minT->lengthOf(); e++) {
                       (*minT)(e) = nd4j::math::nd4j_min((*minT)(e), (*listOfTensors->at(i))(e));
                    }
                }
                else {
                    idx = static_cast<int>((*indices)(i));
                    minT = listOfOutTensors->at(idx);
                    minT->assign(listOfTensors->at(i));
                }
            }
        }
    }

    // segmen mean
    template <typename T>
    void segmentMeanFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = static_cast<int>((*indices)(0));
        if (input->isVector()) {
            T val = T(0.f);
            int count = 0;
            for (int e = 0; e < indices->lengthOf(); e++) {
                if (idx == static_cast<int>((*indices)(e))) {
                   // mean 
                   val += (*input)(e);
                   count++;
                }
                else {
                   (*output)(idx) = val / count;
                    idx = static_cast<int>((*indices)(e));
                    val = (*input)(e);
                    count = 1;
                }
                (*output)(idx) = val / count;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;
            ResultSet<T>* listOfTensors = NDArrayFactory<T>::allTensorsAlongDimension(input, restDims);
            ResultSet<T>* listOfOutTensors = NDArrayFactory<T>::allTensorsAlongDimension(output, restDims);

            int numOfClasses = output->sizeAt(0); // number of classes
            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
            NDArray<T>* meanT = listOfOutTensors->at(idx);
            T count = T(1.f);
            NDArray<T>* meanV = meanT->dup();
            meanV->assign(listOfTensors->at(0));
//#pragma omp parallel for
            for (int i = 1; i < indices->lengthOf(); i++) {
                if (static_cast<int>((*indices)(i)) == idx) {
                    for (int e = 0; e < meanT->lengthOf(); e++) {
                       (*meanV)(e) += (*listOfTensors->at(i))(e);
                    }
                    count += T(1.f);
                }
                else {
                    //meanT->assign(meanV);
                    meanV->template applyScalar<simdOps::Divide<T>>(count, meanT);
                    idx = static_cast<int>((*indices)(i));
                    meanT = listOfOutTensors->at(idx);
                    meanV->assign(listOfTensors->at(i));
                    count = T(1.f);
                }
                meanV->template applyScalar<simdOps::Divide<T>>(count, meanT);
            }
            delete meanV;
            delete listOfTensors;
            delete listOfOutTensors;
        }
    }

    template <typename T>
    void segmentSumFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = static_cast<int>((*indices)(0));
        if (input->isVector()) {
            T val = T(0.f);
            int count = 0;
            for (int e = 0; e < indices->lengthOf(); e++) {
                if (idx == static_cast<int>((*indices)(e))) {
                   // sum 
                   val += (*input)(e);
                }
                else {
                    idx = static_cast<int>((*indices)(e));
                    val = (*input)(e);
                }
                (*output)(idx) = val;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;
            ResultSet<T>* listOfTensors = NDArrayFactory<T>::allTensorsAlongDimension(input, restDims);
            ResultSet<T>* listOfOutTensors = NDArrayFactory<T>::allTensorsAlongDimension(output, restDims);

            int numOfClasses = output->sizeAt(0); // number of classes
            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
            NDArray<T>* sumT = listOfOutTensors->at(idx);

            for (int i = 0; i < indices->lengthOf(); i++) {
                if (static_cast<int>((*indices)(i)) == idx) {
                    for (int e = 0; e < sumT->lengthOf(); e++) {
                       (*sumT)(e) += (*listOfTensors->at(i))(e);
                    }
                }
                else {
                    idx = static_cast<int>((*indices)(i));
                    sumT = listOfOutTensors->at(idx);
                    sumT->assign(listOfTensors->at(i));
                }
            }
            delete listOfTensors;
            delete listOfOutTensors;
        }
    }

    template <typename T>
    void segmentProdFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = static_cast<int>((*indices)(0));
        if (input->isVector()) {
            T val = (*input)(0);
            int count = 0;
            for (int e = 1; e < indices->lengthOf(); e++) {
                if (idx == static_cast<int>((*indices)(e))) {
                   // sum 
                   val *= (*input)(e);
                }
                else {
                    idx = static_cast<int>((*indices)(e));
                    val = (*input)(e);
                }
                (*output)(idx) = val;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;
            ResultSet<T>* listOfTensors = NDArrayFactory<T>::allTensorsAlongDimension(input, restDims);
            ResultSet<T>* listOfOutTensors = NDArrayFactory<T>::allTensorsAlongDimension(output, restDims);

            int numOfClasses = output->sizeAt(0); // number of classes
            NDArray<T>* sumT = listOfOutTensors->at(idx);
            sumT->assign(listOfTensors->at(0));
            for (int i = 1; i < indices->lengthOf(); i++) {
                if (static_cast<int>((*indices)(i)) == idx) {
                    for (int e = 0; e < sumT->lengthOf(); e++) {
                       (*sumT)(e) *= (*listOfTensors->at(i))(e);
                    }
                }
                else {
                    idx = static_cast<int>((*indices)(i));
                    sumT = listOfOutTensors->at(idx);
                    sumT->assign(listOfTensors->at(i));
                }
            }
            delete listOfTensors;
            delete listOfOutTensors;
        }
    }

    template <typename T>
    bool segmentIndicesValidate(NDArray<T>* indices, T& expected, T& output) {
            T val = (*indices)(0);
            for (int e = 1; e < indices->lengthOf(); e++) {
                output = (*indices)(e);
                if (val > output) 
                    return false;
                val = (*indices)(e);
            }
            return true;
    }

    template bool segmentIndicesValidate(NDArray<float>* indices, float& expected, float& output);
    template bool segmentIndicesValidate(NDArray<float16>* indices, float16& expected, float16& output);
    template bool segmentIndicesValidate(NDArray<double>* indices, double& expected, double& output);

    template void segmentMaxFunctor<float>(NDArray<float>* input, NDArray<float>* indices, NDArray<float>* output);
    template void segmentMaxFunctor<float16>(NDArray<float16>* input, NDArray<float16>* , NDArray<float16>* output);
    template void segmentMaxFunctor<double>(NDArray<double>* input, NDArray<double>* , NDArray<double>* output);

    template void segmentMinFunctor<float>(NDArray<float>* input, NDArray<float>* , NDArray<float>* output);
    template void segmentMinFunctor<float16>(NDArray<float16>* input, NDArray<float16>* , NDArray<float16>* output);
    template void segmentMinFunctor<double>(NDArray<double>* input, NDArray<double>* , NDArray<double>* output);

    template void segmentMeanFunctor<float>(NDArray<float>* input, NDArray<float>* , NDArray<float>* output);
    template void segmentMeanFunctor<float16>(NDArray<float16>* input, NDArray<float16>* , NDArray<float16>* output);
    template void segmentMeanFunctor<double>(NDArray<double>* input, NDArray<double>* , NDArray<double>* output);

    template void segmentSumFunctor<float>(NDArray<float>* input, NDArray<float>* , NDArray<float>* output);
    template void segmentSumFunctor<float16>(NDArray<float16>* input, NDArray<float16>* , NDArray<float16>* output);
    template void segmentSumFunctor<double>(NDArray<double>* input, NDArray<double>* , NDArray<double>* output);

    template void segmentProdFunctor<float>(NDArray<float>* input, NDArray<float>* , NDArray<float>* output);
    template void segmentProdFunctor<float16>(NDArray<float16>* input, NDArray<float16>* , NDArray<float16>* output);
    template void segmentProdFunctor<double>(NDArray<double>* input, NDArray<double>* , NDArray<double>* output);

}
}
}