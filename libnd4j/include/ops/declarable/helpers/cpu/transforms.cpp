/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//


#include<ops/declarable/helpers/transforms.h>
#include <array/ResultSet.h>
#include <helpers/ShapeUtils.h>
#include <numeric>
#include <NDArrayFactory.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {



//////////////////////////////////////////////////////////////////////////
void triu(const NDArray& input, NDArray& output, const int diagonal) {

    const int rank = input.rankOf();
    
    switch(rank) {

        case 1:
            for(int i = 0; i < output.sizeAt(0); ++i)
                output({i, i+1, 0,0}).assign(input);
            output.setValueInDiagMatrix(0., diagonal-1, 'l');    
            break;

        case 2:
            output.assign(input);
            output.setValueInDiagMatrix(0., diagonal-1, 'l');    
            break;

        default: 
            auto inTads  = input.allTensorsAlongDimension({rank-2, rank-1});
            auto outTads = output.allTensorsAlongDimension({rank-2, rank-1});

// #pragma omp parallel for schedule(guided) if(inTads->size() > Environment::getInstance()->elementwiseThreshold()) 
            for(int i = 0; i < inTads->size(); ++i) {
                auto inSubArr = inTads->at(i);
                auto outSubArr = outTads->at(i);
                outSubArr->assign(inSubArr);
                outSubArr->setValueInDiagMatrix(0., diagonal-1, 'l');
            }
            delete inTads;
            delete outTads;
    }
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
static void triuBP_(const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal) {

    auto dOdI = NDArrayFactory::_create(&gradO);                // dO/dI
    helpers::triu(input, dOdI, diagonal);

#pragma omp parallel for if(dOdI.lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)     
    for(int i = 0; i < dOdI.lengthOf(); ++i) {
        if(dOdI.getScalar<T>(i) != (T)0.f)
            dOdI.putScalar(i,  T(1.f));
    }

    gradI.assign(dOdI * gradO);                          // chain rule: dLoss/dI = dO/dI * dLoss/dO 
}

    void triuBP(const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal) {
        BUILD_SINGLE_SELECTOR(gradO.dataType(), triuBP_, (input, gradO, gradI, diagonal), LIBND4J_TYPES);
    }


BUILD_SINGLE_TEMPLATE(template void triuBP_, (const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void trace_(const NDArray& input, NDArray& output) {

    const int inRank = input.rankOf();

    auto setOfSubArrs = input.allTensorsAlongDimension({inRank-2, inRank-1});

#pragma omp parallel for if(setOfSubArrs->size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)     
    for(int i = 0; i < setOfSubArrs->size(); ++i)
        output.putScalar(i, setOfSubArrs->at(i)->getTrace());

    delete setOfSubArrs;
}

    void trace(const NDArray& input, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), trace_, (input, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void trace_, (const NDArray& input, NDArray& output), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
template <typename T>
void randomShuffle_(NDArray& input, NDArray& output, nd4j::random::RandomBuffer& rng, const bool isInplace) {

    // check edge cases first
    int temp;
    const int firstDim = input.sizeAt(0);    
    if(input.lengthOf() == 1 || firstDim == 1) {
        
        if(!isInplace)
            output.assign(input);
    } 
    else if (input.isVector() || shape::isLikeVector(input.getShapeInfo(), temp)) {
                
        // apply Fisher-Yates shuffle 
        if(isInplace) {
#pragma omp parallel for if((firstDim-1) > Environment::getInstance()->elementwiseThreshold()) schedule(guided)       
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng.nextInt(0, i);
                if(i == r)
                    continue;
                T _e0 = input.getScalar<T>(i);
                T _e1 = input.getScalar<T>(r);
                //math::nd4j_swap<T>(input(i), input(r));
                input.putScalar<T>(i, _e1);
                input.putScalar<T>(r, _e0);
            }        
        }
        else {        
            std::vector<int> indices(firstDim);        
            std::iota(indices.begin(), indices.end(), 0);        
            output.putScalar<T>(Nd4jLong(0), input.getScalar<T>(0));
#pragma omp parallel for if((firstDim-1) > Environment::getInstance()->elementwiseThreshold()) schedule(guided)       
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng.nextInt(0, i);
                output.putScalar(i, input.getScalar<T>(indices[r]));
                if(i == r)
                    continue;

                output.putScalar(r, input.getScalar<T>(indices[i]));
                math::nd4j_swap<int>(indices[i], indices[r]);
            }           
            rng.rewindH(firstDim-1);
        }
    }
    else {
            
        // evaluate sub-arrays list of input array through all dimensions excluding first one
        std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input.rankOf(), {0});
        auto subArrsListIn = input.allTensorsAlongDimension(dimensions);

        // apply Fisher-Yates shuffle
        if(isInplace) {
#pragma omp parallel for if((firstDim-1) > Environment::getInstance()->elementwiseThreshold()) schedule(guided)        
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng.nextInt(0, i);
                if(i == r)
                    continue;
                subArrsListIn->at(i)->swapUnsafe(*subArrsListIn->at(r));
            }        
        }
        else {
            // evaluate sub-arrays list of output array through all dimensions excluding first one        
            auto subArrsListOut = output.allTensorsAlongDimension(dimensions);
            std::vector<int> indices(firstDim);        
            std::iota(indices.begin(), indices.end(), 0);        
            bool isZeroShuffled = false;
#pragma omp parallel for if((firstDim-1) > Environment::getInstance()->elementwiseThreshold()) schedule(guided)       
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng.nextInt(0, i);
                subArrsListOut->at(i)->assign(subArrsListIn->at(indices[r]));
                if(r == 0)
                    isZeroShuffled = true;
                if(i == r)
                    continue;
                subArrsListOut->at(r)->assign(subArrsListIn->at(indices[i]));
                math::nd4j_swap<int>(indices[i], indices[r]);
            }           
            if(!isZeroShuffled)
                subArrsListOut->at(0)->assign(subArrsListIn->at(0));
            delete subArrsListOut;
        }
        rng.rewindH(firstDim-1);
        delete subArrsListIn;
    }

}

    void randomShuffle(NDArray& input, NDArray& output, nd4j::random::RandomBuffer& rng, const bool isInplace) {
        BUILD_SINGLE_SELECTOR(input.dataType(), randomShuffle_, (input, output, rng, isInplace), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void randomShuffle_, (NDArray& input, NDArray& output, nd4j::random::RandomBuffer& rng, const bool isInplace), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
// initial values of inIdx, outIdx, dim must be equal to zero
template<typename T>
static void recursiveLoopForPad_(const int mode, NDArray& input, const NDArray& paddings, NDArray& output, std::vector<int> dimensions, int dim, int inIdx, int outIdx, NDArray& padValue ) {
    
    int leftOffset;
    // dimensions are array of input dimensions, it is sorted by increasing order
    // every time at the beginning we erase first element from it (not good idea to use vector for this purpose, but luckily it is small enough)
    // then we use this array for tads building, every time while recursion the number of built tads becomes bigger 
    dimensions.erase(dimensions.begin());       
    // build tad basing on output array, also create auxiliary arrays pointing on required output array ranges
    shape::TAD tadOut(output.getShapeInfo(), dimensions.data(), dimensions.size());
    tadOut.createTadOnlyShapeInfo();
    tadOut.createOffsets();
    auto subArrOut = NDArray(output.getBuffer(), tadOut.tadOnlyShapeInfo, output.getWorkspace());
    auto subArr = NDArray(output.getBuffer(), tadOut.tadOnlyShapeInfo, output.getWorkspace());
    // build tad basing on input array, also create auxiliary array pointing on required input array range
    shape::TAD tadIn(input.getShapeInfo(), dimensions.data(), dimensions.size());
    tadIn.createTadOnlyShapeInfo();
    tadIn.createOffsets();
    auto subArrIn = NDArray(input.getBuffer(), tadIn.tadOnlyShapeInfo, output.getWorkspace());
    // these indices take into account recursion and always point to actual tads numbers
    if (input.rankOf() > 1 && output.rankOf() > 1) {// only for non-vector cases
        outIdx = outIdx * output.sizeAt(dim + 1);
        inIdx = inIdx * input.sizeAt(dim + 1);
    }
    // current input tad number, we add to it unity in a loop
    int k = -1;
    // loop through current dimension
    for(int i = 0; i < output.sizeAt(dim); ++i) {
        // corresponds to outer range (relevant indices are absent in input)                        
        leftOffset = paddings.getScalar<int>(dim, 0);
        if(i < leftOffset || i >= (input.sizeAt(dim) + leftOffset))
            continue;

        // increase input tads number
        ++k;
        // recursion condition allows for the fact that tad can't reduce to scalar
        if(dim < input.rankOf() - 2)
            recursiveLoopForPad(mode, input, paddings, output, dimensions, dim + 1, inIdx + k, outIdx + i, padValue);
        else if (paddings.sizeAt(0) > dim + 1){
            leftOffset = paddings.getScalar<int>(dim + 1, 0);
            // shift buffers pointers to actual element position
            if (output.rankOf() > 1) {
                subArrOut.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + i]);
                subArrIn.setBuffer(reinterpret_cast<T*>(input.getBuffer()) + tadIn.tadOffsets[inIdx + i - paddings.getScalar<int>(dim, 0)]);
            }
            else {
                subArrOut.putScalar(i, subArrIn.getScalar<T>(i - leftOffset));
            }
            // most inner loop, corresponds to last dim = rank-1
            switch (mode) {
                case 0:             // CONSTANT mode                    
                    for(int j = 0; j < subArrOut.lengthOf(); ++j)                   
                            if(j < leftOffset || j >= (subArrIn.lengthOf() + leftOffset) )                  // firstly fill with zeros outer ranges
                                subArrOut.putScalar(j, (T)0.f);
                            else
                                subArrOut.putScalar(j, subArrIn.getScalar<T>(j - leftOffset));   // fill middle with elements of input array
                    break;

                case 1:             // REFLECT mode                 
                    for(int j = 1;  j <= leftOffset; ++j)                                               // fill firstly left side 
                        subArrOut.putIndexedScalar(leftOffset - j, subArrIn.getIndexedScalar<T>(j));
                    for(int j = 0; j < subArrIn.lengthOf(); ++j)                                        // fill middle
                        subArrOut.putIndexedScalar(leftOffset + j, subArrIn.getIndexedScalar<T>(j));
                    for(int j = (subArrOut.lengthOf() - leftOffset); j < subArrOut.lengthOf(); ++j)     // fill right side
                        subArrOut.putIndexedScalar(j, subArrIn.getIndexedScalar<T>(subArrOut.lengthOf() - j - 1));
                    break;

                case 2:             // SYMMETRIC mode               
                    for(int j = 1;  j <= leftOffset; ++j)                                               // fill firstly left side 
                        subArrOut.putIndexedScalar(leftOffset - j, subArrIn.getIndexedScalar<T>(j-1));
                    for(int j = 0; j < subArrIn.lengthOf(); ++j)                                        // fill middle
                        subArrOut.putIndexedScalar(leftOffset + j, subArrIn.getIndexedScalar<T>(j));
                    for(int j = (subArrOut.lengthOf() - leftOffset); j < subArrOut.lengthOf(); ++j)     // fill right side
                        subArrOut.putIndexedScalar(j, subArrIn.getIndexedScalar<T>(subArrOut.lengthOf() - j));
                    break;
            }
        }
        else {

             if (mode == 0 && input.rankOf() < 2)
                 subArrOut.putScalar(i, subArrIn.getScalar<T>(i - leftOffset));   // fill middle with elements of input array
        }   
    }   
    // populate sub-array formed previously 
    leftOffset = paddings.getScalar<int>(dim,0);
    switch (mode) {
        case 0:         // CONSTANT mode
            for(int j = 1;  j <= leftOffset; ++j) {
                // fill left side with padValue
                if (output.rankOf() > 1) {
                    subArrOut.setBuffer(
                            reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + leftOffset - j]);
                    subArrOut.assign(padValue);
                }
                else {
                    subArrOut.putScalar(j - 1, padValue);
                }
            }
//            output.printIndexedBuffer("Output at");
            for(int j = (output.sizeAt(dim) - leftOffset); j < output.sizeAt(dim); ++j) {       // fill left side with zeros
                if (output.rankOf() > 1) {
                    subArrOut.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + j]);
                    subArrOut.assign(padValue);
                }
                else {
                    subArrOut.putScalar(j, padValue);
                }
            }
            break;

        case 1:         // REFLECT mode 
            for(int j = 1;  j <= leftOffset; ++j) {                                                     // fill left side 
                subArr.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + leftOffset + j]);
                subArrOut.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + leftOffset - j]);
                subArrOut.assign(&subArr);
            }               
            for(int j = (output.sizeAt(dim) - leftOffset); j < output.sizeAt(dim); ++j) {       // fill right side
                subArr.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + output.sizeAt(dim) + leftOffset - 1 - j]);
                subArrOut.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + j]);
                subArrOut.assign(&subArr);              
            }   
            break;

        case 2:         // SYMMETRIC mode   
            for(int j = 1;  j <= leftOffset; ++j) {                                                     // fill left side
                subArr.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + leftOffset + j - 1]);
                subArrOut.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + leftOffset - j]);
                subArrOut.assign(&subArr);
            }           
            for(int j = (output.sizeAt(dim) - leftOffset); j < output.sizeAt(dim); ++j) {       // fill right side
                subArr.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + output.sizeAt(dim) + leftOffset - j]);
                subArrOut.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + j]);
                subArrOut.assign(&subArr);      
            }
            break;
    }
}

    void recursiveLoopForPad(const int mode, NDArray& input, const NDArray& paddings, NDArray& output, std::vector<int> dimensions, int dim, int inIdx, int outIdx, NDArray& padValue ) {
        BUILD_SINGLE_SELECTOR(input.dataType(), recursiveLoopForPad_, (mode, input, paddings, output, dimensions, dim, inIdx, outIdx, padValue), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void recursiveLoopForPad_, (const int mode, NDArray& input, const NDArray& paddings, NDArray& output, std::vector<int> dimensions, int dim, int inIdx, int outIdx, NDArray& padValue), LIBND4J_TYPES);


////////////////////////////////////////////////////////////////////////
void invertPermutation(const NDArray& input, NDArray& output) {

    std::set<int> uniqueElems;
    const int length = input.lengthOf();    

// #pragma omp parallel for if(length > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
    for(int i = 0; i < length; ++i) {
        
        int elem = input.getScalar<int>(i);
 
        if(!uniqueElems.insert(elem).second)        // this operation forbids us to use #pragma omp
            throw std::runtime_error("helpers::invertPermutation function: input array contains duplicates !");
            
        if(elem < 0 || elem > length - 1)
            throw  std::runtime_error("helpers::invertPermutation function: element of input array is out of range (0, length-1) !");

        output.putScalar<int>(elem, i);
    }
}

////////////////////////////////////////////////////////////////////////
template<typename T>
static void gatherND_(NDArray& input, NDArray& indices, NDArray& output) {

    if (input.ordering() != 'c') 
        input.streamline('c');

    if (indices.ordering() != 'c')
        indices.streamline('c');

    const int rankIn     = input.rankOf();
    const int rankInd    = indices.rankOf();
    const int lastIndDim = indices.sizeAt(-1);
    
    std::vector<int> tadDims(rankIn - lastIndDim);
    std::iota(tadDims.begin(), tadDims.end(), rankInd-1);
    auto innerMostOut = output.allTensorsAlongDimension(tadDims);

    auto innerMostInd = indices.allTensorsAlongDimension({rankInd-1});
    
    std::iota(tadDims.begin(), tadDims.end(), lastIndDim);
    auto innerMostIn = input.allTensorsAlongDimension(tadDims);

    Nd4jLong* outerShapeInfo = nullptr;
    ALLOCATE(outerShapeInfo, input.getWorkspace(), shape::shapeInfoLength(lastIndDim), Nd4jLong);
    outerShapeInfo[0] = lastIndDim;
    for(int i = 1; i <= lastIndDim; ++i)
        outerShapeInfo[i] = input.sizeAt(i-1);
    shape::updateStrides(outerShapeInfo, input.ordering());

    Nd4jLong idx[MAX_RANK];

    for(int i = 0; i < innerMostInd->size(); ++i) {
                
        auto idxSubArr = innerMostInd->at(i);
        
        for(int j = 0; j < lastIndDim; ++j) {
            if(idxSubArr->getScalar<Nd4jLong>(j) >= input.sizeAt(j))
                throw std::runtime_error("helpers::gatherND function: indices array contains wrong elements, each element must be smaller than corresponding dimension of input array !");
            idx[j] = idxSubArr->getScalar<Nd4jLong>(j);
        }
                
        auto currentInd0 = shape::getOffset(0, shape::shapeOf(outerShapeInfo), shape::stride(outerShapeInfo), idx, lastIndDim);

        if(rankIn != lastIndDim) {
            auto outSubArr = innerMostOut->at(i);
            outSubArr->assign(innerMostIn->at(currentInd0));
        }
        else
            output.putScalar(i, input.getScalar<T>(currentInd0));
    }

    delete innerMostInd;
    delete innerMostIn;
    delete innerMostOut;
    RELEASE(outerShapeInfo, input.getWorkspace());    
}

    void gatherND(NDArray& input, NDArray& indices, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), gatherND_, (input, indices, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void gatherND_, (NDArray& input, NDArray& indices, NDArray& output), LIBND4J_TYPES);


////////////////////////////////////////////////////////////////////////
template<typename T>
static void gather_(NDArray* input, const NDArray* indices, NDArray* output, const std::vector<int>& intArgs) {

    int axis = intArgs.size() > 0 ? intArgs[0] : 0;
    const int inputRank = input->rankOf();
    if(axis < 0)
        axis += inputRank;

    const int numOfIntArgs = intArgs.size();

    if (indices != nullptr) {        

        for(int i = 0; i < indices->lengthOf(); ++i)
            if(indices->getScalar<Nd4jLong>(i) >= input->sizeAt(axis))
                throw std::runtime_error("helpers::gather function: indices array contains wrong elements, each element must be smaller than corresponding dimension of input array !");
    
        // first case: indices consist of only one scalar
        if(indices->isScalar()) {
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {axis});
            shape::TAD tad(input->getShapeInfo(), dimensions.data(), dimensions.size());
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();
            auto tadArr = NDArray(reinterpret_cast<void *>(reinterpret_cast<T*>(input->getBuffer()) + tad.tadOffsets[indices->getScalar<Nd4jLong>(0)]), tad.tadOnlyShapeInfo, output->getWorkspace());
            output->assign(&tadArr);
        }
        else if (input->rankOf() == 1 && indices->isVector()) {
            // special case
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)     
            for (int e = 0; e < indices->lengthOf(); e++)
                output->putScalar(e, input->getScalar<T>(indices->getScalar<Nd4jLong>(e)));
        }
        // second case: indices is vector
        else if(indices->isVector()) {      
            auto listOut = output->allTensorsAlongDimension(ShapeUtils::evalDimsToExclude(output->rankOf(), {axis}));
            auto listIn  = input->allTensorsAlongDimension(ShapeUtils::evalDimsToExclude(input->rankOf(),  {axis}));
#pragma omp parallel for if(listOut->size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)             
            for(int i = 0; i < listOut->size(); ++i)
                listOut->at(i)->assign(listIn->at(indices->getScalar<Nd4jLong>(i)));
            delete listOut;
            delete listIn;
        }
        // third case: indices is usual n-dim array
        else {
            std::vector<int> dimsOut(indices->rankOf());
            std::iota(dimsOut.begin(), dimsOut.end(), axis);   // fill with axis, axis+1, ... indices->rankOf()-1
            std::vector<int> temp1 = ShapeUtils::evalDimsToExclude(output->rankOf(), dimsOut);
            std::vector<int> temp2 = ShapeUtils::evalDimsToExclude(input->rankOf(),  {axis});
            auto listOut = output->allTensorsAlongDimension(temp1);
            auto listIn = input->allTensorsAlongDimension(temp2 );
#pragma omp parallel for if(listOut->size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
            for(int i = 0; i < listOut->size(); ++i)
                listOut->at(i)->assign(listIn->at(indices->getScalar<Nd4jLong>(i)));
            delete listOut;
            delete listIn;
        }
    } 
    else {          // in this case always (numOfIntArgs > 1) !!!
        
        for(int i = 1; i < numOfIntArgs; ++i)
            if(intArgs[i] >= input->sizeAt(axis))
                throw std::runtime_error("helpers::gather function: some of input indexes is larger than corresponding shape of input array !");

        // we only allow scalar/vector case here
        if (numOfIntArgs == 2) {
            // scalar case
            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {axis});
            shape::TAD tad(input->getShapeInfo(), dimensions.data(), dimensions.size());
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();
            auto tadArr = NDArray(reinterpret_cast<void *>(reinterpret_cast<T*>(input->getBuffer()) + tad.tadOffsets[intArgs[1]]), tad.tadOnlyShapeInfo);
            output->assign(&tadArr);
        } else {
            // vector case
            auto listOut = output->allTensorsAlongDimension(ShapeUtils::evalDimsToExclude(output->rankOf(), {axis}));
            auto listIn  = input->allTensorsAlongDimension(ShapeUtils::evalDimsToExclude(input->rankOf(),  {axis}));

            // that's fine, since we know that number of iArgs matches number of elements in listOut
#pragma omp parallel for if(listOut->size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)     
            for(int i = 0; i < listOut->size(); ++i)
                listOut->at(i)->assign(listIn->at(intArgs[i+1]));
            delete listOut;
            delete listIn;
        }
    }    
}

    void gather(NDArray* input, const NDArray* indices, NDArray* output, const std::vector<int>& intArgs) {
        BUILD_SINGLE_SELECTOR(input->dataType(), gather_, (input, indices, output, intArgs), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void gather_, (NDArray* input, const NDArray* indices, NDArray* output, const std::vector<int>& intArgs), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
void eye(NDArray& output) {

    const int rank = output.rankOf();
    auto arrs = output.allTensorsAlongDimension({rank-2, rank-1});

#pragma omp parallel for if(arrs->size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
    for(int i = 0; i < arrs->size(); ++i)
        arrs->at(i)->setIdentity();
    
    delete arrs;
}

//////////////////////////////////////////////////////////////////////////
void scatterUpdate(NDArray& operand, NDArray& updates, const std::vector<int>* intArgs) {

    int opCode = (*intArgs)[0];
    int dimSize = (*intArgs)[1];    
    unsigned long e;
    unsigned long limg = 2 + dimSize;
    std::vector<int> tadDimension(limg-2);
    for (e = 2; e < limg; e++)
        tadDimension[e-2] = (*intArgs)[e];

    // increasing counter to skip numIndices
    e++;
    std::vector<int> indices;
    std::vector<int> indicesU;
    int cnt = 0;
    for (; e < intArgs->size(); e++) {
        indices.push_back((*intArgs)[e]);
        indicesU.push_back(cnt++);
    }

    std::unique_ptr<ResultSet> tadsOperand(operand.multipleTensorsAlongDimension(indices, tadDimension));
    std::unique_ptr<ResultSet> tadsUpdate(updates.multipleTensorsAlongDimension(indicesU, tadDimension));

#pragma omp parallel for if(indices.size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided) proc_bind(close) shared(tadsOperand, tadsUpdate)
    for (unsigned long x = 0; x < indices.size(); x++) {
                
        auto tad = tadsOperand->at(x);
        auto tadUpdates = tadsUpdate->at(x);

        if (tad->lengthOf() != tadUpdates->lengthOf())
            continue;

        switch (opCode) {
            case 0:
                tad->applyPairwiseTransform(pairwise::Add, tadUpdates, tad, nullptr);
                break;
            case 1:
                tad->applyPairwiseTransform(pairwise::Subtract, tadUpdates, tad, nullptr);
                break;
            case 2:
                tad->applyPairwiseTransform(pairwise::Multiply, tadUpdates, tad, nullptr);
                break;
            case 3:
                tad->applyPairwiseTransform(pairwise::Divide, tadUpdates, tad, nullptr);
                break;
            case 4:
                tad->applyPairwiseTransform(pairwise::ReverseSubtract, tadUpdates, tad, nullptr);
                break;
            case 5:
                tad->applyPairwiseTransform(pairwise::ReverseDivide, tadUpdates, tad, nullptr);
                break;
            case 6:
                tad->applyPairwiseTransform(pairwise::Copy, tadUpdates, tad, nullptr);
                break;
            default:
                continue;                 
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
static void mergeMaxIndex_(const std::vector<NDArray*>& inArrs, NDArray& output) {

    const Nd4jLong numArgs = inArrs.size();
    auto x = inArrs[0];

#pragma omp parallel for if(x->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
    for (Nd4jLong e = 0; e < x->lengthOf(); e++) {
        T max = -DataTypeUtils::max<T>();
        Nd4jLong idx = 0;
            
        for (int i = 0; i < numArgs; i++){
            
            T v = inArrs[i]->getScalar<T>(e);
            if (v > max) {
                max = v;
                idx = i;
            }
        }
        output.putScalar(e, idx);
    }
}
    void mergeMaxIndex(const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_SINGLE_SELECTOR(inArrs[0]->dataType(), mergeMaxIndex_, (inArrs, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void mergeMaxIndex_, (const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
template<typename T>
static void mergeMax_(const std::vector<NDArray*>& inArrs, NDArray& output) {
    
    const Nd4jLong numArgs = inArrs.size();
    auto x = inArrs[0];

#pragma omp parallel for if(x->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided) proc_bind(close)
     for (Nd4jLong e = 0; e < x->lengthOf(); e++) {
        T max = -DataTypeUtils::max<T>();
        for (int i = 0; i < numArgs; i++) {
            T v = inArrs[i]->getScalar<T>(e);
            if (v > max)
                max = v;
        }
        output.putScalar(e, max);
    }
}
    void mergeMax(const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_SINGLE_SELECTOR(output.dataType(), mergeMax_, (inArrs, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void mergeMax_, (const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
template<typename T>
static void mergeAvg_(const std::vector<NDArray*>& inArrs, NDArray& output) {
    
    const Nd4jLong numArgs = inArrs.size();
    const T factor = 1.f / numArgs;
    auto x = inArrs[0];
        
#pragma omp parallel for if(x->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided) proc_bind(close)
    for (Nd4jLong e = 0; e < x->lengthOf(); e++) {
        T sum = 0.;
        for (int i = 0; i < numArgs; i++) { 
            T v = inArrs[i]->getScalar<T>(e);
            sum += v;
        }
        output.putScalar<T>(e, sum * factor);
    }
}
    void mergeAvg(const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_SINGLE_SELECTOR(output.dataType(), mergeAvg_, (inArrs, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void mergeAvg_, (const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
template<typename T>
static void mergeAdd_(const std::vector<NDArray*>& inArrs, NDArray& output) {
    
    const Nd4jLong numArgs = inArrs.size();
    auto x = inArrs[0];
        
#pragma omp parallel for if(x->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided) proc_bind(close)
    for (Nd4jLong e = 0; e < x->lengthOf(); e++) {
        
        T sum = (T) 0.f;
        
        for (int i = 0; i < numArgs; i++) 
            sum += inArrs[i]->getScalar<T>(e);

        output.putScalar(e, sum);
    }
}
    void mergeAdd(const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_SINGLE_SELECTOR(output.dataType(), mergeAdd_, (inArrs, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void mergeAdd_, (const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
template<typename T>
static void clipByNorm_(NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {
        
    const int rank = input.rankOf();
   auto norm2 = input.reduceAlongDims(reduce::Norm2, dimensions);

    if (isInplace) {
        if(norm2.lengthOf() == 1) {

            if(norm2.getScalar<T>(0) > clipNorm.getScalar<T>(0))
                input *= (clipNorm.getScalar<T>(0) / norm2.getScalar<T>(0));
        }
        else {

            std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(rank, dimensions);
            const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(input.getShapeInfo(), dimsToExclude);

#pragma omp parallel for schedule(guided) 
            for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {
                if (norm2.getScalar<T>(i) > clipNorm.getScalar<T>(0)) {
                    
                    auto inputSubArr  = input(i, dimsToExclude);
                    inputSubArr *= (clipNorm.getScalar<T>(0) / norm2.getScalar<T>(i));
                }
            }
        }
    }
    else {
        
        if(norm2.lengthOf() == 1) {

            if(norm2.getScalar<T>(0) > clipNorm.getScalar<T>(0))
                output.assign( input * (clipNorm / norm2.getScalar<T>(0)));
            else
                output.assign( input );
        }
        else {
            
            std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(rank, dimensions);
            const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(input.getShapeInfo(), dimsToExclude);
            std::vector<Nd4jLong> idxRanges(rank * 2);

#pragma omp parallel for schedule(guided) firstprivate(idxRanges)
            for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {

                ShapeUtils::evalIdxRangesForSubArr(i, input.getShapeInfo(), dimsToExclude, idxRanges.data());

                auto outputSubArr = output(idxRanges);
                auto inputSubArr  = input(idxRanges);
                outputSubArr.assign(inputSubArr);
                
                if (norm2.getScalar<T>(i) > clipNorm.getScalar<T>(0))
                    outputSubArr *= clipNorm / norm2.getScalar<T>(i);
            }           
        }
    }
}

    void clipByNorm(NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {
        BUILD_SINGLE_SELECTOR(output.dataType(), clipByNorm_, (input, output, dimensions, clipNorm, isInplace), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByNorm_, (NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace), FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
template<typename T>
static void clipByNormBP_(const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const NDArray& clipNorm) {
    
    const int rank = input.rankOf();

    auto norm2 = input.reduceAlongDims(reduce::Norm2, dimensions);

    if(norm2.lengthOf() == 1) {        

        const T N = norm2.getScalar<T>(0);

        auto cn = clipNorm.getScalar<T>(0);
        
        if(N > cn) {

            const T sumOfProd = (input * gradO).reduceNumber(reduce::Sum).getScalar<T>(0);    // reduce to scalar
            const T factor1 = static_cast<T>(1.f) / N;
            const T factor3 = factor1 / (N * N) ;                                            // 1 / (N*N*N)

            auto lambda = LAMBDA_TT(elem1, elem2, cn, sumOfProd, factor1, factor3) {
                return cn * (factor1 * elem2 - factor3 * elem1 * sumOfProd);
            };

            (const_cast<NDArray&>(input)).applyPairwiseLambda<T>(&gradO, lambda, &gradI);
        }
        else 
            gradI.assign(gradO);
    }
    else {
            
        std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(rank, dimensions);
        const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(input.getShapeInfo(), dimsToExclude);
        std::vector<Nd4jLong> idxRanges(rank * 2);

#pragma omp parallel for schedule(guided) firstprivate(idxRanges)
        for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {

            ShapeUtils::evalIdxRangesForSubArr(i, input.getShapeInfo(), dimsToExclude, idxRanges.data());
            T N = norm2.getScalar<T>(i);

            auto gradOSubArr = gradO(idxRanges);
            auto gradISubArr = gradI(idxRanges);

            auto cn = clipNorm.getScalar<T>(0);

            if (N > cn) {
                
                auto inputSubArr = input(idxRanges);
                
                const T sumOfProd = (inputSubArr * gradOSubArr).reduceNumber(reduce::Sum).getScalar<T>(0);    // reduce to scalar
                const T factor1 = static_cast<T>(1.f) / N;
                const T factor3 = factor1 / (N * N) ;                                            // 1 / (N*N*N)

                auto lambda = LAMBDA_TT(elem1, elem2, cn, sumOfProd, factor1, factor3) {
                    return cn * (factor1 * elem2 - factor3 * elem1 * sumOfProd);
                };
                inputSubArr.applyPairwiseLambda<T>(&gradOSubArr, lambda, &gradISubArr);
            }
            else
                gradISubArr.assign(gradOSubArr);
        }           
    }
}

    void clipByNormBP(const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const NDArray& clipNorm) {
        BUILD_SINGLE_SELECTOR(gradI.dataType(), clipByNormBP_, (input, gradO, gradI, dimensions, clipNorm), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByNormBP_, (const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const NDArray& clipNorm), FLOAT_TYPES);


//////////////////////////////////////////////////////////////////////////
template<typename T>
static void clipByAveraged_(NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {

    auto cn = clipNorm.getScalar<T>(0);
    if (dimensions.size() == 0) {
        // all-reduce
        T n2 = input.reduceNumber(reduce::Norm2).getScalar<T>(0) / input.lengthOf();
        if (n2 <= cn) {
            if (!isInplace)
                output.assign(input);
        } 
        else {
            const T factor = cn / n2;
            auto lambda = LAMBDA_T(_x, factor) { return _x * factor; };
            input.applyLambda<T>(lambda, &output);
        }
    } 
    else {
        // along dimension
        auto norm2 = input.reduceAlongDims(reduce::Norm2, dimensions, false);
        if (!isInplace)
                output.assign(input);
        auto tads = output.allTensorsAlongDimension(dimensions);
        // TODO: make this CUDA-compliant somehow
        for (int e = 0; e < tads->size(); e++) {
            T n2 = norm2.getScalar<T>(e) / tads->at(e)->lengthOf();
            const T factor = cn / n2;
            if (n2 > cn) {
                auto lambda = LAMBDA_T(_x, factor) {return _x * factor;};
                tads->at(e)->applyLambda<T>(lambda, &output);
            }
        }
        delete tads;
    }
}

    void clipByAveraged(NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {
        BUILD_SINGLE_SELECTOR(input.dataType(), clipByAveraged_, (input, output, dimensions, clipNorm, isInplace), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByAveraged_, (NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace), FLOAT_TYPES);



//////////////////////////////////////////////////////////////////////////
template<typename T>
static void mirrorPad_(const NDArray& input, const NDArray& paddings, NDArray& output, const int mode) {
    
    // mode:  0 - REFLECT, else - SYMMETRIC
    const int reflBorder = (bool)mode ? 1 : 0;
    const int symmBorder = (bool)mode ? 0 : 1;

    const int rank        = input.rankOf();
    const Nd4jLong outLen = output.lengthOf();
    const Nd4jLong inLen  = input.lengthOf();    

    if(rank <= 1) {

        const auto leftSide  = paddings.getScalar<Nd4jLong>(0);
        const auto rightSide = paddings.getScalar<Nd4jLong>(1);

#pragma omp parallel for if(outLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(int i = 0; i < outLen; ++i) {
            
            for(int j = 0; j < leftSide; ++j)
                output.putScalar(j, input.getScalar<T>(inLen - leftSide + symmBorder - j));
            for(int j = 0; j < inLen; ++j)
                output.putScalar(j + leftSide, input.getScalar<T>(j));
            for(int j = 0; j < rightSide; ++j)
                output.putScalar(leftSide + inLen + j, input.getScalar<T>(inLen - 1 - symmBorder - j));
        }  
    }
    else {

        std::vector<Nd4jLong> inIdx(rank), outIdx(rank);
#pragma omp parallel for if(outLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided) firstprivate(inIdx, outIdx)
        for(int i = 0; i < outLen; ++i) {

            shape::ind2subC(rank, output.shapeOf(), i, outIdx.data());

            for(int j = 0; j < rank; ++j) {
            
                const auto leftSide  = paddings.getScalar<T>(j, 0);

                if(outIdx[j] < leftSide) 
                    inIdx[j] = leftSide - outIdx[j] - reflBorder;

                else if(outIdx[j] >= leftSide && outIdx[j] < leftSide + input.sizeAt(j)) 
                    inIdx[j] = outIdx[j] - leftSide;

                else
                    inIdx[j] = 2 * input.sizeAt(j) + leftSide - outIdx[j] - 1 - symmBorder;                
            }
    
            auto outOffset = shape::getOffset(0, output.shapeOf(), output.stridesOf(), outIdx.data(), rank);
            auto inOffset  = shape::getOffset(0, input.shapeOf(),  input.stridesOf(),  inIdx.data(),  rank);
            reinterpret_cast<T*>(output.buffer())[outOffset] = reinterpret_cast<T*>(input.getBuffer())[inOffset];
        }
    }
}

    void mirrorPad(const NDArray& input, const NDArray& paddings, NDArray& output, const int mode) {
        BUILD_SINGLE_SELECTOR(input.dataType(), mirrorPad_, (input, paddings, output, mode), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void mirrorPad_, (const NDArray& input, const NDArray& paddings, NDArray& output, const int mode), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
template<typename T>
static void concat_(const std::vector<NDArray*>& inArrs, NDArray& output, const int axis) {

    const int numOfArrs = inArrs.size();
    bool allC = true;
    bool allScalar = true;
    bool allVectors = true;
    
    Nd4jLong lenOfFirstArr = inArrs[0]->lengthOf();

    //detect whether all arrays are c ordered or not
    //Also detect whether they are all scalars
    for(int i = 0; i < numOfArrs; i++) {
        allC &= (inArrs[i]->ordering() == 'c');
        allScalar &= (inArrs[i]->isScalar());
        allVectors &= (inArrs[i]->isRowVector() && inArrs[0]->lengthOf() == lenOfFirstArr);
    }

    //we are merging all scalars
    if(allScalar) {
        for(int i = 0; i < numOfArrs; i++)
            reinterpret_cast<T*>(output.getBuffer())[i] = reinterpret_cast<T*>(inArrs[i]->getBuffer())[0];
        return;
    }

    if(allC && axis == 0 && allVectors && output.ordering() == 'c') {
        
        if (numOfArrs >= 8) {

#pragma omp parallel for schedule(guided)
            for (int r = 0; r < numOfArrs; r++) {

                T *z = reinterpret_cast<T*>(output.getBuffer()) + (r * lenOfFirstArr);
                T *x = reinterpret_cast<T*>(inArrs[r]->getBuffer());

#pragma omp simd
                for (Nd4jLong e = 0; e < lenOfFirstArr; e++)
                    z[e] = x[e];
            }
        } 
        else {
            int currBuffer = 0;
            int currBufferOffset = 0;
            for (int i = 0; i < output.lengthOf(); i++) {
                reinterpret_cast<T*>(output.getBuffer())[i] = reinterpret_cast<T*>(inArrs[currBuffer]->getBuffer())[currBufferOffset++];
                if (currBufferOffset >= inArrs[currBuffer]->lengthOf()) {
                    currBuffer++;
                    currBufferOffset = 0;
                }
            }
        }
        return;
    }
    
    const int rank  = inArrs[0]->rankOf();
    const int rank2 = 2*rank;
    std::vector<std::vector<Nd4jLong>> indices(numOfArrs, std::vector<Nd4jLong>(rank2,0));

    // take into account indices for first array
    indices[0][2 * axis + 1] = inArrs[0]->sizeAt(axis);

    // loop through the rest of input arrays
    for(int i = 1; i < numOfArrs; ++i) {
        indices[i][2 * axis]     = indices[i-1][2 * axis + 1];                                // index start from
        indices[i][2 * axis + 1] = indices[i-1][2 * axis + 1] + inArrs[i]->sizeAt(axis);      // index end with (excluding)
    }

// #pragma omp parallel for if(numOfArrs > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
#pragma omp parallel for schedule(guided)
    for(int i = 0; i < numOfArrs; ++i) {
        auto temp = output(indices[i], true);
        temp.assign(inArrs[i]);
    }
}

    void concat(const std::vector<NDArray*>& inArrs, NDArray& output, const int axis) {
        BUILD_SINGLE_SELECTOR(output.dataType(), concat_,(inArrs, output, axis), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void concat_, (const std::vector<NDArray*>& inArrs, NDArray& output, const int axis), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void tileBP_(const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps) {

    T* gradIBuff      = reinterpret_cast<T*>(gradI.getBuffer());
    const T* gradOBuff      = reinterpret_cast<T*>(gradO.getBuffer());
    const Nd4jLong gradILen = gradI.lengthOf();
    const Nd4jLong gradOLen = gradO.lengthOf();  // gradOLen >= gradILen
    const Nd4jLong gradIEWS = nd4j::math::nd4j_abs<Nd4jLong>(gradI.ews());
    const Nd4jLong gradOEWS = gradO.ews();

    // initial zeroing of gradI content
    if(gradIEWS == 1)
        memset(gradIBuff, 0, gradILen * sizeof(T));
    else
#pragma omp parallel for schedule(static) proc_bind(close)
        for (int i = 0; i < gradILen * gradIEWS; i += gradIEWS)
            gradIBuff[i] = static_cast<T>(0.f);


    if(gradO.ordering() == 'c' && gradOEWS == 1) {
#pragma omp parallel for simd if(gradOLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(Nd4jLong i=0;  i<gradOLen; ++i) {
            auto idx = shape::subArrayIndex(gradO.getShapeInfo(), gradI.getShapeInfo(), i);
            gradI.putScalar(idx, gradI.getScalar<T>(idx) + gradOBuff[i]);
        }
    }
    else if(gradO.ordering() == 'c' && gradOEWS > 1) {
#pragma omp parallel for simd if(gradOLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(Nd4jLong i=0;  i<gradOLen; ++i) {
            auto idx = shape::subArrayIndex(gradO.getShapeInfo(), gradI.getShapeInfo(), i);
            gradI.putScalar(idx, gradI.getScalar<T>(idx) + gradOBuff[i * gradOEWS]);
        }
    }
    else {
        Nd4jLong idx[MAX_RANK];
        Nd4jLong* gradOShape   = gradO.shapeOf();
        Nd4jLong* gradOStrides = gradO.stridesOf();
        const int gradORank    = gradO.rankOf();
#pragma omp parallel for simd if(gradOLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided) private(idx)
        for(Nd4jLong i=0;  i<gradOLen; ++i) {
            shape::ind2subC(gradORank, gradOShape, i, gradOLen, idx);
            auto fidx = shape::subArrayIndex(gradO.getShapeInfo(), gradI.getShapeInfo(), i);
            gradI.putScalar(fidx, gradI.getScalar<T>(fidx) + gradOBuff[shape::getOffset(0, gradOShape, gradOStrides, idx, gradORank)]);
        }
    }
}

    void tileBP(const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps) {
        BUILD_SINGLE_SELECTOR(gradI.dataType(), tileBP_, (gradO, gradI, reps), FLOAT_TYPES);
    }


    BUILD_SINGLE_TEMPLATE(template void tileBP_, (const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps), FLOAT_TYPES);

}
}
}