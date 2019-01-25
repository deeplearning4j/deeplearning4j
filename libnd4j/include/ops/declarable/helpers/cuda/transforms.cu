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
#include <helpers/TAD.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {

    //////////////////////////////////////////////////////////////////////////
    void triu(graph::LaunchContext* context, const NDArray& input, NDArray& output, const int diagonal) {

    }


    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    static void triuBP_(graph::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal) {

    }

    void triuBP(graph::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal) {
        BUILD_SINGLE_SELECTOR(gradO.dataType(), triuBP_, (context, input, gradO, gradI, diagonal), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void triuBP_, (graph::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal), LIBND4J_TYPES);

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    static void trace_(graph::LaunchContext* context, const NDArray& input, NDArray& output) {

    }

    void trace(graph::LaunchContext* context, const NDArray& input, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), trace_, (context, input, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void trace_, (graph::LaunchContext* context, const NDArray& input, NDArray& output), LIBND4J_TYPES);

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    void randomShuffle_(graph::LaunchContext* context, NDArray& input, NDArray& output, nd4j::random::RandomBuffer& rng, const bool isInplace) {

    }

    void randomShuffle(graph::LaunchContext* context, NDArray& input, NDArray& output, nd4j::random::RandomBuffer& rng, const bool isInplace) {
        BUILD_SINGLE_SELECTOR(input.dataType(), randomShuffle_, (context, input, output, rng, isInplace), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void randomShuffle_, (graph::LaunchContext* context, NDArray& input, NDArray& output, nd4j::random::RandomBuffer& rng, const bool isInplace), LIBND4J_TYPES);

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    void pad_(graph::LaunchContext* context, const int mode, const NDArray& input, const NDArray& paddings, NDArray& output, NDArray const& padValue) {

    }

    void pad(graph::LaunchContext* context, const int mode, const NDArray& input, const NDArray& paddings, NDArray& output, NDArray const& padValue) {
        BUILD_SINGLE_SELECTOR(input.dataType(), pad_, (context, mode, input, paddings, output, padValue), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void pad_, (graph::LaunchContext* context, const int mode, const NDArray& input, const NDArray& paddings, NDArray& output, NDArray const& padValue), LIBND4J_TYPES);

    ////////////////////////////////////////////////////////////////////////
    void invertPermutation(graph::LaunchContext* context, const NDArray& input, NDArray& output) {

    }

    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void gatherND_(graph::LaunchContext* context, NDArray& input, NDArray& indices, NDArray& output) {

    }

    void gatherND(graph::LaunchContext* context, NDArray& input, NDArray& indices, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), gatherND_, (context, input, indices, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void gatherND_, (graph::LaunchContext* context, NDArray& input, NDArray& indices, NDArray& output), LIBND4J_TYPES);


    ////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void gather_(graph::LaunchContext* context, NDArray* input, const NDArray* indices, NDArray* output, const std::vector<int>& intArgs) {

    }

    void gather(graph::LaunchContext* context, NDArray* input, const NDArray* indices, NDArray* output, const std::vector<int>& intArgs) {
        BUILD_SINGLE_SELECTOR(input->dataType(), gather_, (context, input, indices, output, intArgs), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void gather_, (graph::LaunchContext* context, NDArray* input, const NDArray* indices, NDArray* output, const std::vector<int>& intArgs), LIBND4J_TYPES);

    //////////////////////////////////////////////////////////////////////////
    void eye(graph::LaunchContext* context, NDArray& output) {

    }

    //////////////////////////////////////////////////////////////////////////
    void scatterUpdate(graph::LaunchContext* context, NDArray& operand, NDArray& updates, const std::vector<int>* intArgs) {

    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void mergeMaxIndex_(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {

    }

    void mergeMaxIndex(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_SINGLE_SELECTOR(inArrs[0]->dataType(), mergeMaxIndex_, (context, inArrs, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void mergeMaxIndex_, (graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void mergeMax_(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {

    }

    void mergeMax(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_SINGLE_SELECTOR(output.dataType(), mergeMax_, (context, inArrs, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void mergeMax_, (graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void mergeAvg_(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {

    }

    void mergeAvg(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_SINGLE_SELECTOR(output.dataType(), mergeAvg_, (context, inArrs, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void mergeAvg_, (graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void mergeAdd_(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {

    }

    void mergeAdd(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_SINGLE_SELECTOR(output.dataType(), mergeAdd_, (context, inArrs, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void mergeAdd_, (graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output), LIBND4J_TYPES);

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void clipByNorm_(graph::LaunchContext* context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {

    }

    void clipByNorm(graph::LaunchContext* context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {
        BUILD_SINGLE_SELECTOR(output.dataType(), clipByNorm_, (context, input, output, dimensions, clipNorm, isInplace), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByNorm_, (graph::LaunchContext* context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace), FLOAT_TYPES);

    template <typename T>
    static void clipByGlobalNorm_(graph::LaunchContext* context, std::vector<NDArray*> const& inputs, double clipNorm, nd4j::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {

    }

    void clipByGlobalNorm(graph::LaunchContext* context, std::vector<NDArray*> const& inputs, double clipNorm, nd4j::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {
        BUILD_SINGLE_SELECTOR(outputs[0]->dataType(), clipByGlobalNorm_, (context, inputs, clipNorm, workspace, outputs, isInplace), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByGlobalNorm_, (graph::LaunchContext* context, std::vector<NDArray*> const& inputs, double clipNorm, nd4j::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace), FLOAT_TYPES);

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void clipByNormBP_(graph::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const NDArray& clipNorm) {

    }

    void clipByNormBP(graph::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const NDArray& clipNorm) {
        BUILD_SINGLE_SELECTOR(gradI.dataType(), clipByNormBP_, (context, input, gradO, gradI, dimensions, clipNorm), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByNormBP_, (graph::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const NDArray& clipNorm), FLOAT_TYPES);


    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void clipByAveraged_(graph::LaunchContext* context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {

    }

    void clipByAveraged(graph::LaunchContext* context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {
        BUILD_SINGLE_SELECTOR(input.dataType(), clipByAveraged_, (context, input, output, dimensions, clipNorm, isInplace), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByAveraged_, (graph::LaunchContext* context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace), FLOAT_TYPES);

/*
    if (d1 > params[1])
    return params[1];
    else if (d1 < params[0])
    return params[0];
    else return d1;
*/

    template <typename T>
    static void clipByValue_(graph::LaunchContext* context, NDArray& input, double leftBound, double rightBound, NDArray& output) {

    }

    void clipByValue(graph::LaunchContext* context, NDArray& input, double leftBound, double rightBound, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), clipByValue_, (context, input, leftBound, rightBound, output), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByValue_, (graph::LaunchContext* context, NDArray& input, double leftBound, double rightBound, NDArray& output);, FLOAT_TYPES);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void mirrorPad_(graph::LaunchContext* context, const NDArray& input, const NDArray& paddings, NDArray& output, const int mode) {

    }

    void mirrorPad(graph::LaunchContext* context, const NDArray& input, const NDArray& paddings, NDArray& output, const int mode) {
        BUILD_SINGLE_SELECTOR(input.dataType(), mirrorPad_, (context, input, paddings, output, mode), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void mirrorPad_, (graph::LaunchContext* context, const NDArray& input, const NDArray& paddings, NDArray& output, const int mode), LIBND4J_TYPES);

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void concat_(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output, const int axis) {

    }

    void concat(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output, const int axis) {
        BUILD_SINGLE_SELECTOR(output.dataType(), concat_,(context, inArrs, output, axis), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void concat_, (graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output, const int axis), LIBND4J_TYPES);

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    static void tileBP_(graph::LaunchContext* context, const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps) {

    }

    void tileBP(graph::LaunchContext* context, const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps) {
        BUILD_SINGLE_SELECTOR(gradI.dataType(), tileBP_, (context, gradO, gradI, reps), FLOAT_TYPES);
    }


    BUILD_SINGLE_TEMPLATE(template void tileBP_, (graph::LaunchContext* context, const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps), FLOAT_TYPES);

}
}
}
