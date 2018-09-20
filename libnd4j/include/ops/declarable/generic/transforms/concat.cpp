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
//  @author raver119@gmail.com
//  @author Yurii Shyrma (iuriish@yahoo.com)
//

#include<ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>
#include<array>

namespace nd4j {
    namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(concat, -1, 1, false, 0, 1) {

    // first of all take into account possible presence of empty arrays
    // also if scalar is present -> copy its value to vector with length=1
    std::vector<NDArray*> nonEmptyArrs;
    std::vector<int> arrsToDelete;
    int index = 0;
    for(int i = 0; i < block.width(); ++i) {
        
        if(!INPUT_VARIABLE(i)->isEmpty()) {
            
            if(INPUT_VARIABLE(i)->rankOf() == 0) {
                auto vec = new NDArray('c', {1}, INPUT_VARIABLE(0)->dataType(), block.getWorkspace());
                (*vec) = *INPUT_VARIABLE(i);
                nonEmptyArrs.push_back(vec);
                arrsToDelete.push_back(index);
            }
            else{
                nonEmptyArrs.push_back(INPUT_VARIABLE(i));
            }
            ++index;
        }
    }
    
    const int numOfArrs = nonEmptyArrs.size();    
    REQUIRE_TRUE(numOfArrs > 0, 0, "CONCAT op: at least one input array must be non-empty!");

    auto output = OUTPUT_VARIABLE(0);
    
    const int rank = nonEmptyArrs[0]->rankOf();     //  look up to first non-empty array

    int axis = INT_ARG(0);
    if(axis < 0)
        axis += rank;

    // ******** input validation ******** //
    REQUIRE_TRUE(0 <= axis && axis < rank, 0, "CONCAT op: input axis must be in range [0, %i], but got %i instead!", rank-1, axis);    

    for(int i = 1; i < numOfArrs; ++i)        
        REQUIRE_TRUE(nonEmptyArrs[i]->rankOf() == rank, 0, "CONCAT op: all input arrays must have the same rank !");

    for(int i = 1; i < numOfArrs; ++i) {        
        for(int dim = 0; dim < rank; ++dim)
            if(dim != axis)                
                REQUIRE_TRUE(nonEmptyArrs[i]->sizeAt(dim) == nonEmptyArrs[0]->sizeAt(dim), 0, "CONCAT op: all input arrays must have the same dimensions (except those on input axis) !");
    }
    // ******** end of input validation ******** //

    if(numOfArrs == 1) 
        output->assign(nonEmptyArrs[0]);
    else 
        helpers::concat(nonEmptyArrs, *output, axis);

    // delete dynamically allocated vectors with length=1
    for(int index : arrsToDelete)
        delete nonEmptyArrs[index];

    return Status::OK();
}

    DECLARE_SYN(ParallelConcat, concat);
    DECLARE_SYN(concat_v2, concat);
    DECLARE_SYN(concatv2, concat);

DECLARE_SHAPE_FN(concat) {
    
    // first of all take into account possible presence of empty arrays
    // also if scalar is present -> use the shape of vector with length=1 instead 
    std::vector<Nd4jLong*> nonEmptyArrShapes;
    std::vector<int> shapesToDelete;
    int index = 0;
    for(int i = 0; i < block.width(); ++i) {
        
        if(!INPUT_VARIABLE(i)->isEmpty()) {
            
            if(inputShape->at(i)[0] == 0) {
                Nd4jLong* vecShapeInfo = nullptr;
                ALLOCATE(vecShapeInfo, block.getWorkspace(), shape::shapeInfoLength(1), Nd4jLong);
                shape::shapeVector(1, vecShapeInfo);
                nonEmptyArrShapes.push_back(vecShapeInfo);
                shapesToDelete.push_back(index);
            }
            else{
                nonEmptyArrShapes.push_back(inputShape->at(i));
            }
            ++index;
        }
    }

    const int numOfArrs = nonEmptyArrShapes.size();    
    REQUIRE_TRUE(numOfArrs > 0, 0, "CONCAT op: at least one input array must be non-empty!");    
    
    const int rank = nonEmptyArrShapes[0][0];     //  look up to first non-empty array

    int axis = INT_ARG(0);
    if(axis < 0)
        axis += rank;

    // ******** input validation ******** //
    REQUIRE_TRUE(0 <= axis && axis < rank, 0, "CONCAT op: input axis must be in range [0, %i], but got %i instead!", rank-1, axis);

    for(int i = 1; i < numOfArrs; ++i)        
        REQUIRE_TRUE(nonEmptyArrShapes[i][0] == rank, 0, "CONCAT op: all input arrays must have the same rank !");

    for(int i = 1; i < numOfArrs; ++i) {        
        for(int dim = 0; dim < rank; ++dim)
            if(dim != axis)                
                REQUIRE_TRUE(nonEmptyArrShapes[i][dim+1] == nonEmptyArrShapes[0][dim+1], 0, "CONCAT op: all input arrays must have the same dimensions (except those on input axis) !");
    }
    // ******** end of input validation ******** //
    

    Nd4jLong* outShapeInfo(nullptr);
    COPY_SHAPE(nonEmptyArrShapes[0], outShapeInfo);
    
    // case when we have only one input array
    if(numOfArrs == 1) {                
        shape::updateStrides(outShapeInfo, shape::order(nonEmptyArrShapes[0]));
        return SHAPELIST(outShapeInfo);
    }

    for(int i = 1; i < numOfArrs; ++i)
        outShapeInfo[axis + 1] += nonEmptyArrShapes[i][axis + 1];

    shape::updateStrides(outShapeInfo, shape::order(nonEmptyArrShapes[0]));

    // delete dynamically allocated vectors shapes with length=1
    for(int index : shapesToDelete)        
        RELEASE(nonEmptyArrShapes[index], block.getWorkspace());

    return SHAPELIST(outShapeInfo);
}


        // //////////////////////////////////////////////////////////////////////////
        // CUSTOM_OP_IMPL(concat, -1, 1, false, 0, -2){
        //     // do something here{
        //     NDArray<T> *last = INPUT_VARIABLE((int) block.width() - 1);

        //     int _dimension = 0;
        //     if (block.numI() > 0)
        //         _dimension = INT_ARG(0);
        //     else {
        //         _dimension = (int) last->getScalar(0);
        //     }

        //     // we want to ensure that all
        //     NDArray<T> *first = nullptr;
        //     auto output = OUTPUT_VARIABLE(0);

        //     int elements = 0;

        //     for (int e = 0; e < block.width(); e++) {
        //         auto arr = INPUT_VARIABLE(e);
        //         if (!arr->isEmpty())
        //             elements++;

        //         // we must find first non-empty element here
        //         if (!arr->isEmpty() && first == nullptr)
        //             first = arr;
        //     }

        //     REQUIRE_TRUE(first != nullptr, 0, "Concat: at least 1 non-empty input required!");

        //     // it's possible to get into situation when your input has only 1 input. That's just assign
        //     if (elements == 1) {
        //         output->assign(first);
        //         return Status::OK();
        //     }

        //     bool oldScalars = first->rankOf() == 2 && first->isScalar();

        //     auto buffers = new Nd4jPointer[elements];
        //     auto shapes = new Nd4jPointer[elements];

        //     buffers[0] = (Nd4jPointer) first->getBuffer();
        //     shapes[0] = (Nd4jPointer) first->getShapeInfo();

        //     if (_dimension < 0)
        //         _dimension += first->rankOf();

        //     if (nd4j::Environment::getInstance()->isDebugAndVerbose()) {
        //         printf("Shape %i: ", 0);
        //         shape::printShapeInfoLinear((Nd4jLong *) shapes[0]);
        //     }

        //     int er = 0;
        //     for (int e = 0; e < block.width(); e++) {
        //         Variable<T> *var = block.variable(e);
        //         auto array = var->getNDArray();

        //         if (array->isEmpty())
        //             continue;

        //         buffers[er] = reinterpret_cast<Nd4jPointer>(array->getBuffer());
        //         shapes[er++] = reinterpret_cast<Nd4jPointer>(array->getShapeInfo());

        //         oldScalars &= array->rankOf() == 2 && array->isScalar();

        //         if (nd4j::Environment::getInstance()->isDebugAndVerbose()) {
        //             printf("Shape %i: ", e);
        //             shape::printShapeInfoLinear(array->shapeInfo());
        //         }
        //     }
        //     if (nd4j::Environment::getInstance()->isDebugAndVerbose())
        //         fflush(stdout);

        //     if (oldScalars) {
        //         nd4j_debug("OLD_SCALARS!\n","");
        //         _dimension = 1;
        //     }

        //     nd4j::SpecialMethods<T>::concatCpuGeneric(_dimension, elements, buffers, shapes, output->getBuffer(), output->getShapeInfo());

        //     STORE_RESULT(*output);

        //     if (nd4j::Environment::getInstance()->isDebugAndVerbose())
        //         output->printShapeInfo("Concat result shape");

        //     delete[] buffers;
        //     delete[] shapes;

        //     return ND4J_STATUS_OK;
        // }

        // DECLARE_SYN(ParallelConcat, concat);
        // DECLARE_SYN(concat_v2, concat);
        // DECLARE_SYN(concatv2, concat);
        
        // DECLARE_SHAPE_FN(concat) {
        //     auto inp = inputShape->at(0);
        //     int _dimension = INT_ARG(0);

        //     NDArray<T>* first = nullptr;
        //     auto last = inputShape->at(inputShape->size() - 1);

        //     Nd4jLong elements = 0;
        //     Nd4jLong *newShape;

        //     for (int  e = 0; e < inputShape->size(); e++) {
        //         auto s = INPUT_VARIABLE(e);

        //         if (!s->isEmpty()) {
        //             elements++;

        //             if (first == nullptr)
        //                 first = s;
        //         }
        //     }


        //     { // special cases for 0D concat
        //         bool allScalars = true;
        //         bool hasScalars = false;
        //         for (int e = 0; e < block.width(); e++) {
        //             auto c = INPUT_VARIABLE(e);

        //             if (c->isEmpty())
        //                 continue;

        //             allScalars &= c->rankOf() == 0;
        //             hasScalars |= c->rankOf() == 0;
        //         }

        //         // all scalars
        //         if (allScalars) {
        //             ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(1), Nd4jLong);

        //             shape::shapeBuffer(1, &elements, newShape);
        //             return SHAPELIST(newShape);
        //         }

        //         // any scalar
        //         if (hasScalars) {
        //             ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(1), Nd4jLong);
        //             Nd4jLong length = shape::length(inp);
        //             for (int i = 1; i < block.width(); i++) {
        //                 auto c = INPUT_VARIABLE(i);
        //                 if (c->isEmpty())
        //                     continue;

        //                 length += c->lengthOf();
        //             }

        //             shape::shapeBuffer(1, &length, newShape);
        //             return SHAPELIST(newShape);
        //         }
        //     }

            
        //     ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(first->shapeInfo()), Nd4jLong);

        //     if (_dimension < 0)
        //         _dimension += first->rankOf();

        //     std::memcpy(newShape, first->shapeInfo(), shape::shapeInfoByteLength(first->shapeInfo()));
        //     for (int i = 0; i < inputShape->size(); i++) {
        //         auto s = INPUT_VARIABLE(i);

        //         // FIXME: s == first is bad, but fast. alternatively we can subtract first size out of result
        //         if (s->isEmpty() || s == first)
        //             continue;

        //         newShape[_dimension + 1] += shape::shapeOf(inputShape->at(i))[_dimension];
        //     }

        //     shape::updateStrides(newShape, first->ordering());

        //     return SHAPELIST(newShape);
        // }


        CUSTOM_OP_IMPL(concat_bp, -1, -1, false, 0, 1) {
            auto epsilonNext = INPUT_VARIABLE(block.width() - 1);

            auto first = INPUT_VARIABLE(0);

            int axis = INT_ARG(0);

            if (axis < 0)
                axis += first->rankOf();

            int startPos = 0;
            for (int e = 0; e < block.width() - 1; e++) {
                auto originalChunk = INPUT_VARIABLE(e);
                auto epsilonChunk = OUTPUT_VARIABLE(e);
                IndicesList indices;

                int width = originalChunk->sizeAt(axis);            

                for (int e = 0; e < epsilonNext->rankOf(); e++) {
                    if (e == axis)
                        indices.push_back(NDIndex::interval(startPos, startPos + width));
                    else
                        indices.push_back(NDIndex::all());
                }

                auto subarray = epsilonNext->subarray(indices);
                epsilonChunk->assign(subarray);

                startPos += width;

                delete subarray;
            }

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(concat_bp) {
            auto shapeList = SHAPELIST();

            for (int e = 0; e < inputShape->size() - 1; e++) {
                auto inShape = inputShape->at(e);
                Nd4jLong* newShape;
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inShape), Nd4jLong);
                memcpy(newShape, inShape, shape::shapeInfoByteLength(inShape));

                shapeList->push_back(newShape);
            }

            return shapeList;
        }
    }
}
