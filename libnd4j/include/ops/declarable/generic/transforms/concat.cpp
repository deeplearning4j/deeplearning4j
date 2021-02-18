/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

namespace sd  {
namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(concat, -1, 1, false, 0, 0) {

    REQUIRE_TRUE(block.width() > 0, 0, "CONCAT op: No input arrays were provided");

    const bool isAxisInLastArr = block.getBArguments()->size() == 0 ? false : B_ARG(0);

    const int numOfInArrs = isAxisInLastArr ? block.width() - 1 : block.width();

    // first of all take into account possible presence of empty arrays
    // also if scalar is present -> copy its value to vector with length=1
    std::vector<const NDArray*> nonEmptyArrs;
    std::vector<int> arrsToDelete;
    int index = 0;
    bool allOfSameType = true;
    auto rankOfFirstArr = block.width() > 0 ? INPUT_VARIABLE(0)->rankOf() : 0;
    auto typeOfFirstArr = block.width() > 0 ? INPUT_VARIABLE(0)->dataType() : block.dataType();

    for(int i = 0; i < numOfInArrs; ++i) {
        auto input = INPUT_VARIABLE(i);
        auto currentRank = input->rankOf();

// TODO: follow two lines are in accordance to current tf.concat spec. Commented for compatibility with legacy
//        REQUIRE_TRUE(currentRank > 0, 0, "Rank of input variable %i must be greater 0, but is %lld instead.", i, currentRank);
//        REQUIRE_TRUE(rankOfFirstArr == currentRank, 0, "Number of dimensions in concat should be equals, but for %i input variable %lld != %lld appears.", i, currentRank, rankOfFirstArr);
        if(!input->isEmpty()) {

            allOfSameType &= (typeOfFirstArr == input->dataType());

            if(input->rankOf() == 0) {
                auto vec = new NDArray('c', {1}, input->dataType(), block.launchContext());
                vec->assign(input);
                nonEmptyArrs.push_back(vec);
                arrsToDelete.push_back(index);
            }
            else{
                nonEmptyArrs.push_back(input);
            }
            ++index;
        }
    }

    const int numOfNonEmptyArrs = nonEmptyArrs.size();

    if(numOfNonEmptyArrs == 0){
        //All inputs are empty arrays -> return empty, mainly for TF import compatibility (no op)
        REQUIRE_TRUE(OUTPUT_VARIABLE(0)->isEmpty(), 0, "CONCAT op: If all input variables are empty, output must be empty");
        return Status::OK();
    }

    const int rank = nonEmptyArrs[0]->rankOf();                     //  look up to first non-empty array
    int axis = isAxisInLastArr ? INPUT_VARIABLE(block.width() - 1)->e<int>(0) : INT_ARG(0);
    if(axis < 0){
        axis += rank;
    }

    // ******** input validation ******** //
    REQUIRE_TRUE(allOfSameType, 0, "CONCAT op: all of input arrays must have same type !");
    REQUIRE_TRUE(nonEmptyArrs[0]->dataType() == OUTPUT_VARIABLE(0)->dataType(), 0, "CONCAT op: output array should have the same type as inputs arrays !");
    REQUIRE_TRUE(0 <= axis && (axis < rank || (axis == 0 && rank == 0)), 0, "CONCAT op: input axis must be in range [0, %i], but got %i instead!", rank-1, axis);

    for(int i = 1; i < numOfNonEmptyArrs; ++i)
        REQUIRE_TRUE(nonEmptyArrs[i]->rankOf() == rank, 0, "CONCAT op: all input arrays must have the same rank !");

    for(int i = 1; i < numOfNonEmptyArrs; ++i) {
        for(int dim = 0; dim < rank; ++dim)
            if(dim != axis)
                REQUIRE_TRUE(nonEmptyArrs[i]->sizeAt(dim) == nonEmptyArrs[0]->sizeAt(dim), 0, "CONCAT op: all input arrays must have the same dimensions (except those on input axis) !");
    }
    // ******** end of input validation ******** //

    auto output = OUTPUT_VARIABLE(0);

    if(numOfNonEmptyArrs == 1)
        output->assign(nonEmptyArrs[0]);
    else
        helpers::concat(block.launchContext(), nonEmptyArrs, *output, axis);

    // delete dynamically allocated vectors with length=1
    for(int index : arrsToDelete)
        delete nonEmptyArrs[index];

    return Status::OK();
}

    DECLARE_SYN(ParallelConcat, concat);
    DECLARE_SYN(concat_v2, concat);
    DECLARE_SYN(concatv2, concat);

        DECLARE_TYPES(concat) {
            getOpDescriptor()
                    ->setAllowedInputTypes(sd::DataType::ANY);
                    // ->setSameMode(true);
        }

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(concat) {

    REQUIRE_TRUE(block.width() > 0, 0, "CONCAT op: No input arrays were provided");

    const bool isAxisInLastArr = block.getBArguments()->size() == 0 ? false : B_ARG(0);

    const int numOfInArrs = isAxisInLastArr ? block.width() - 1 : block.width();

    // first of all take into account possible presence of empty arrays
    // also if scalar is present -> use the shape of vector with length=1 instead
    ShapeList arrShapes;
    std::vector<int> shapesToDelete;
    int index = 0;
    for(int i = 0; i < numOfInArrs; ++i) {

        if(inputShape->at(i)[0] == 0) {
            if (shape::isEmpty(inputShape->at(i)))
                arrShapes.push_back(ConstantShapeHelper::getInstance().vectorShapeInfo(0, INPUT_VARIABLE(0)->dataType()));
            else
                arrShapes.push_back(ConstantShapeHelper::getInstance().vectorShapeInfo(1, INPUT_VARIABLE(0)->dataType()));
        }
        else{
            arrShapes.push_back(inputShape->at(i));
        }
        ++index;
    }

    const int numOfNonEmptyArrs = arrShapes.size();

    const int rank = shape::rank(arrShapes.at(0));

    int axis = isAxisInLastArr ? INPUT_VARIABLE(block.width() - 1)->e<int>(0) : INT_ARG(0);
    if(axis < 0){
        axis += rank;
    }

    // ******** input validation ******** //
    REQUIRE_TRUE(0 <= axis && axis < rank, 0, "CONCAT op: input axis must be in range [0, %i], but got %i instead!", rank-1, axis);

    for(int i = 1; i < numOfNonEmptyArrs; ++i)
        REQUIRE_TRUE(shape::rank(arrShapes.at(i)) == rank, 0, "CONCAT op: all input arrays must have the same rank !");

    for(int i = 1; i < numOfNonEmptyArrs; ++i) {
        for(int dim = 0; dim < rank; ++dim)
            if(dim != axis)
                REQUIRE_TRUE(arrShapes.at(i)[dim+1] == arrShapes.at(0)[dim+1], 0, "CONCAT op: all input arrays must have the same dimensions (except those on input axis) !");
    }
    // ******** end of input validation ******** //


    Nd4jLong* outShapeInfo(nullptr);
    COPY_SHAPE(arrShapes.at(0), outShapeInfo);

    // case when we have only one input array
    if(numOfNonEmptyArrs == 1) {
        ShapeUtils::updateStridesAndType(outShapeInfo, arrShapes.at(0), shape::order(arrShapes.at(0)));
        return SHAPELIST(CONSTANT(outShapeInfo));
    }

    for(int i = 1; i < numOfNonEmptyArrs; ++i)
        outShapeInfo[axis + 1] += arrShapes.at(i)[axis + 1];

    ShapeUtils::updateStridesAndType(outShapeInfo, arrShapes.at(0), shape::order(arrShapes.at(0)));

    // delete dynamically allocated vectors shapes with length=1
//    for(int index : shapesToDelete)
//        RELEASE(arrShapes[index], block.getWorkspace());

    auto result = ConstantShapeHelper::getInstance().createShapeInfo(ShapeDescriptor(outShapeInfo));
    RELEASE(outShapeInfo, block.getWorkspace());
    return SHAPELIST(result);
}


        // //////////////////////////////////////////////////////////////////////////
        // CUSTOM_OP_IMPL(concat, -1, 1, false, 0, -2){
        //     // do something here{
        //     NDArray<T> *last = INPUT_VARIABLE((int) block.width() - 1);

        //     int _dimension = 0;
        //     if (block.numI() > 0)
        //         _dimension = INT_ARG(0);
        //     else {
        //         _dimension = (int) last->e(0);
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

        //     buffers[0] = (Nd4jPointer) first->buffer();
        //     shapes[0] = (Nd4jPointer) first->shapeInfo();

        //     if (_dimension < 0)
        //         _dimension += first->rankOf();

        //     if (sd::Environment::getInstance().isDebugAndVerbose()) {
        //         printf("Shape %i: ", 0);
        //         shape::printShapeInfoLinear((Nd4jLong *) shapes[0]);
        //     }

        //     int er = 0;
        //     for (int e = 0; e < block.width(); e++) {
        //         Variable<T> *var = block.variable(e);
        //         auto array = var->getNDArray();

        //         if (array->isEmpty())
        //             continue;

        //         buffers[er] = reinterpret_cast<Nd4jPointer>(array->buffer());
        //         shapes[er++] = reinterpret_cast<Nd4jPointer>(array->shapeInfo());

        //         oldScalars &= array->rankOf() == 2 && array->isScalar();

        //         if (sd::Environment::getInstance().isDebugAndVerbose()) {
        //             printf("Shape %i: ", e);
        //             shape::printShapeInfoLinear(array->shapeInfo());
        //         }
        //     }
        //     if (sd::Environment::getInstance().isDebugAndVerbose())
        //         fflush(stdout);

        //     if (oldScalars) {
        //         nd4j_debug("OLD_SCALARS!\n","");
        //         _dimension = 1;
        //     }

        //     sd::SpecialMethods<T>::concatCpuGeneric(_dimension, elements, buffers, shapes, output->buffer(), output->shapeInfo());

        //     STORE_RESULT(*output);

        //     if (sd::Environment::getInstance().isDebugAndVerbose())
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

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(concat_bp, -1, -1, false, 0, 0) {

    const bool isAxisInLastArr = block.getBArguments()->size() == 0 ? false : B_ARG(0);

    const int numOfInArrs = isAxisInLastArr ? block.width() - 1 : block.width();

    auto epsilonNext = INPUT_VARIABLE(numOfInArrs - 1);

    auto first = INPUT_VARIABLE(0);

    const int axis = isAxisInLastArr ? INPUT_VARIABLE(block.width() - 1)->e<int>(0) : (INT_ARG(0) >= 0 ? INT_ARG(0) : INT_ARG(0) + INPUT_VARIABLE(0)->rankOf());

    int startPos = 0;

    for (int e = 0; e < numOfInArrs - 1; e++) {
        auto originalChunk = INPUT_VARIABLE(e);
        auto epsilonChunk = OUTPUT_VARIABLE(e);
        std::vector<Nd4jLong> indices(2 * epsilonNext->rankOf());

        int width = originalChunk->sizeAt(axis);

        for (int e = 0; e < epsilonNext->rankOf(); e++) {
            if (e == axis)
                indices[2*e + 1] = (indices[2*e] = startPos) + width;
            else
                indices[2*e + 1] = indices[2*e] = 0;
        }

        auto subarray = (*epsilonNext)(indices, true);
        epsilonChunk->assign(subarray);

        startPos += width;
    }

    return ND4J_STATUS_OK;
}

DECLARE_TYPES(concat_bp) {
    getOpDescriptor()
            ->setAllowedInputTypes(sd::DataType::ANY)
            ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(concat_bp) {

    const bool isAxisInLastArr = block.getBArguments()->size() == 0 ? false : B_ARG(0);

    const int numOfInArrs = isAxisInLastArr ? block.width() - 1 : block.width();

    auto shapeList = SHAPELIST();

    for (int e = 0; e < numOfInArrs - 1; e++) {
        auto inShape = inputShape->at(e);
        shapeList->push_back(ConstantShapeHelper::getInstance().createShapeInfo(ShapeDescriptor(ArrayOptions::dataType(inShape), shape::order(inShape), shape::shapeOf(inShape), shape::rank(inShape))));
    }

    return shapeList;
}


}
}
