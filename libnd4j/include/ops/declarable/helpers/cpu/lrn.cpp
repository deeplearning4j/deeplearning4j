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
//

#include <ops/declarable/helpers/lrn.h>
#include <Status.h>

namespace nd4j {
namespace ops {
namespace helpers {

#ifdef HAVE_MKLDNN
using namespace mkldnn;

static void getMKLDNNMemoryDescLrn(const NDArray* src, const NDArray* diff_src, const NDArray* dst,
        mkldnn::memory::desc* lrn_src_md, mkldnn::memory::desc* lrn_diff_src_md, mkldnn::memory::desc* lrn_dst_md,
        mkldnn::memory::desc* user_src_md, mkldnn::memory::desc* user_diff_src_md, mkldnn::memory::desc* user_dst_md, int axis) {
    const Nd4jLong* shape = src->getShapeInfo();
    long rank = shape[0];
    long dim1 = axis; // MKL-DNN supports only 1 axis, which has to be the "channel" one
    long dim2 = axis >= 2 ? 1 : 2;
    long dim3 = axis >= 3 ? 2 : 3;
    mkldnn::memory::dims lrn_src_tz = { (int)shape[1], (int)shape[dim1 + 1], rank > 2 ? (int)shape[dim2 + 1] : 1, rank > 3 ? (int)shape[dim3 + 1] : 1};

    auto type = mkldnn::memory::data_type::f32;
    auto format = axis == 1 ? mkldnn::memory::format::nchw : mkldnn::memory::format::nhwc;
    auto supposed_to_be_any_format = format; // doesn't work with "any"

    if (src != nullptr && src->getBuffer() != nullptr && lrn_src_md != nullptr) {
        *lrn_src_md = mkldnn::memory::desc({ lrn_src_tz }, type, supposed_to_be_any_format);
        *user_src_md = mkldnn::memory::desc({ lrn_src_tz }, type, format);
        user_src_md->data.format = mkldnn_blocked;
        user_src_md->data.layout_desc.blocking.strides[0][0] = src->stridesOf()[0];
        user_src_md->data.layout_desc.blocking.strides[0][1] = src->stridesOf()[dim1];
        user_src_md->data.layout_desc.blocking.strides[0][2] = rank > 2 ? src->stridesOf()[dim2] : 1;
        user_src_md->data.layout_desc.blocking.strides[0][3] = rank > 3 ? src->stridesOf()[dim3] : 1;
    }

    if (diff_src != nullptr && diff_src->getBuffer() != nullptr && lrn_diff_src_md != nullptr) {
        *lrn_diff_src_md = mkldnn::memory::desc({ lrn_src_tz }, type, supposed_to_be_any_format);
        *user_diff_src_md = mkldnn::memory::desc({ lrn_src_tz }, type, format);
        user_diff_src_md->data.format = mkldnn_blocked;
        user_diff_src_md->data.layout_desc.blocking.strides[0][0] = diff_src->stridesOf()[0];
        user_diff_src_md->data.layout_desc.blocking.strides[0][1] = diff_src->stridesOf()[dim1];
        user_diff_src_md->data.layout_desc.blocking.strides[0][2] = rank > 2 ? diff_src->stridesOf()[dim2] : 1;
        user_diff_src_md->data.layout_desc.blocking.strides[0][3] = rank > 3 ? diff_src->stridesOf()[dim3] : 1;
    }

    if (dst != nullptr && dst->getBuffer() != nullptr && lrn_dst_md != nullptr) {
        *lrn_dst_md = mkldnn::memory::desc({ lrn_src_tz }, type, supposed_to_be_any_format);
        *user_dst_md = mkldnn::memory::desc({ lrn_src_tz }, type, format);
        user_dst_md->data.format = mkldnn_blocked;
        user_dst_md->data.layout_desc.blocking.strides[0][0] = dst->stridesOf()[0];
        user_dst_md->data.layout_desc.blocking.strides[0][1] = dst->stridesOf()[dim1];
        user_dst_md->data.layout_desc.blocking.strides[0][2] = rank > 2 ? dst->stridesOf()[dim2] : 1;
        user_dst_md->data.layout_desc.blocking.strides[0][3] = rank > 3 ? dst->stridesOf()[dim3] : 1;
    }
}
#endif

    template <typename T>
    static int lrnFunctor_(nd4j::graph::Context& block, NDArray* input, NDArray* output, int depth, float bias, float alpha, float beta) {

        int totalLength = input->lengthOf();
        int lastDim = input->sizeAt(-1);
        int chunkCount = totalLength / lastDim;
        T* inputBuffer = reinterpret_cast<T*>(input->buffer());
        T* outputBuffer = reinterpret_cast<T*>(output->buffer());
#ifdef HAVE_MKLDNN
    if (block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported({input, output})) {
        std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
        if (streams.empty()) {
            streams.push_back(MKLDNNStream("lrn"));
        }

        if (streams[0].checkAndReset({input}, {output}, {(float)bias, (float)alpha, (float)beta}, {depth})) {
            mkldnn_memory_desc_t empty;
            mkldnn::memory::desc lrn_src_md(empty), lrn_dst_md(empty), user_src_md(empty), user_dst_md(empty);

            getMKLDNNMemoryDescLrn(input, nullptr, output, &lrn_src_md, nullptr, &lrn_dst_md, &user_src_md, nullptr, &user_dst_md, input->rankOf() - 1);

            auto lrn_desc = lrn_forward::desc(prop_kind::forward_inference, lrn_across_channels, lrn_src_md, (2 * depth + 1), alpha * (2 * depth + 1), beta, bias);

            auto engine = streams[0].getEngine();
            auto lrn_prim_desc = lrn_forward::primitive_desc(lrn_desc, engine);
            auto user_src_memory = mkldnn::memory({user_src_md, engine}, input->buffer());
            auto user_dst_memory = mkldnn::memory({user_dst_md, engine}, output->buffer());

            auto lrn_src_memory = user_src_memory;
            streams[0].addMemory(user_src_memory);
            if (mkldnn::memory::primitive_desc(lrn_prim_desc.src_primitive_desc())
                    != user_src_memory.get_primitive_desc()) {
                lrn_src_memory = mkldnn::memory(lrn_prim_desc.src_primitive_desc());
                streams[0].addMemory(lrn_src_memory);
                streams[0].addOperation(reorder(user_src_memory, lrn_src_memory));
            }

            auto lrn_dst_memory = user_dst_memory;
            streams[0].addMemory(user_dst_memory);
            if (mkldnn::memory::primitive_desc(lrn_prim_desc.dst_primitive_desc())
                    != user_dst_memory.get_primitive_desc()) {
                lrn_dst_memory = mkldnn::memory(lrn_prim_desc.dst_primitive_desc());
                streams[0].addMemory(lrn_dst_memory);
            }

            streams[0].addOperation(lrn_forward(lrn_prim_desc, lrn_src_memory, lrn_dst_memory));

            if (mkldnn::memory::primitive_desc(lrn_prim_desc.dst_primitive_desc())
                    != user_dst_memory.get_primitive_desc()) {
                streams[0].addOperation(reorder(lrn_dst_memory, user_dst_memory));
            }
        }

        streams[0].submitAndWait();
        return ND4J_STATUS_OK;
    }
#endif
    nd4j_debug("MKL-DNN is not used for lrn!\n", 0);

        const T tbias = static_cast<T>(bias);
        const T tbeta = static_cast<T>(beta);

        if (output->ews() == 1 && input->ews() == 1 && input->ordering() == 'c' && output->ordering() == 'c') {

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (int c = 0; c < chunkCount; c++) {
                const int shift = c * lastDim;
                auto iX = inputBuffer + shift;
                T quadSum = 0.f;

                for (int e = 0; e < lastDim; e++) {
                    const int begin = nd4j::math::nd4j_max<int>(0, e - depth);
                    const int end = nd4j::math::nd4j_min<int>(depth + e + 1, lastDim);

                    if (begin == 0) {
                        // at the beginning of rolling window we always read everything
                        quadSum = 0;
                        for (int pos = begin; pos < end; ++pos) {
                            T val = iX[pos];
                            quadSum += val * val;
                        }
                   } else if (end == lastDim) {
                        // at the end of the window we do the same
                        quadSum = 0;
                        for (int pos = begin; pos < end; ++pos) {
                            T val = iX[pos];
                            quadSum += val * val;
                        }
                    } else {
                        // at any other window we add last value and subtract previous last value
                        T prev = iX[begin - 1];
                        T val = iX[end];
                        quadSum += val * val;
                        quadSum -= prev * prev;
                    }

                    T dividor = nd4j::math::nd4j_pow<T, T, T>(tbias + alpha * quadSum, tbeta);
                    outputBuffer[shift + e] = iX[e] / dividor;
                }
            }
        } else {

            PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
            for (int c = 0; c < chunkCount; c++) {
                for (int e = 0; e < lastDim; e++) {
                    int begin = nd4j::math::nd4j_max(0, e - depth);
                    int end = nd4j::math::nd4j_min(depth + e + 1, lastDim);
                    T quadSum = 0;
                    int shift = c * lastDim;

                    PRAGMA_OMP_SIMD_SUM(quadSum)
                    for (int pos = begin; pos < end; ++pos) {
                        T val = inputBuffer[shape::getIndexOffset(shift + pos, input->getShapeInfo(), input->lengthOf())];
                        quadSum += val * val;
                    }

                    T dividor = nd4j::math::nd4j_pow<T, T, T>(bias + alpha * quadSum, beta);
                    outputBuffer[shape::getIndexOffset(shift + e, output->shapeInfo(), output->lengthOf())] = inputBuffer[shape::getIndexOffset(shift + e, input->getShapeInfo(), input->lengthOf())] / dividor;

                }
            }
        }

        return Status::OK();
    }

    template <typename T>
    static int lrnFunctorEx_(nd4j::graph::Context& block, NDArray* input, NDArray* output, NDArray* scale, int depth, float bias, float alpha, float beta) {

        int totalLength = input->lengthOf();
        int lastDim = input->sizeAt(-1);
        int chunkCount = totalLength / lastDim;
        T* inputBuffer = reinterpret_cast<T*>(input->buffer());
        T* outputBuffer = reinterpret_cast<T*>(output->buffer());
#ifdef HAVE_MKLDNN
            if (block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported({input, output})) {
        std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
        if (streams.empty()) {
            streams.push_back(MKLDNNStream("lrn"));
        }

        if (streams[0].checkAndReset({input}, {output}, {(float)bias, (float)alpha, (float)beta}, {depth})) {
            mkldnn_memory_desc_t empty;
            mkldnn::memory::desc lrn_src_md(empty), lrn_dst_md(empty), user_src_md(empty), user_dst_md(empty);

            getMKLDNNMemoryDescLrn(input, nullptr, output, &lrn_src_md, nullptr, &lrn_dst_md, &user_src_md, nullptr, &user_dst_md, input->rankOf() - 1);

            auto lrn_desc = lrn_forward::desc(prop_kind::forward_inference, lrn_across_channels, lrn_src_md, (2 * depth + 1), alpha * (2 * depth + 1), beta, bias);

            auto engine = streams[0].getEngine();
            auto lrn_prim_desc = lrn_forward::primitive_desc(lrn_desc, engine);
            auto user_src_memory = mkldnn::memory({user_src_md, engine}, input->buffer());
            auto user_dst_memory = mkldnn::memory({user_dst_md, engine}, output->buffer());

            auto lrn_src_memory = user_src_memory;
            streams[0].addMemory(user_src_memory);
            if (mkldnn::memory::primitive_desc(lrn_prim_desc.src_primitive_desc())
                    != user_src_memory.get_primitive_desc()) {
                lrn_src_memory = mkldnn::memory(lrn_prim_desc.src_primitive_desc());
                streams[0].addMemory(lrn_src_memory);
                streams[0].addOperation(reorder(user_src_memory, lrn_src_memory));
            }

            auto lrn_dst_memory = user_dst_memory;
            streams[0].addMemory(user_dst_memory);
            if (mkldnn::memory::primitive_desc(lrn_prim_desc.dst_primitive_desc())
                    != user_dst_memory.get_primitive_desc()) {
                lrn_dst_memory = mkldnn::memory(lrn_prim_desc.dst_primitive_desc());
                streams[0].addMemory(lrn_dst_memory);
            }

            streams[0].addOperation(lrn_forward(lrn_prim_desc, lrn_src_memory, lrn_dst_memory));

            if (mkldnn::memory::primitive_desc(lrn_prim_desc.dst_primitive_desc())
                    != user_dst_memory.get_primitive_desc()) {
                streams[0].addOperation(reorder(lrn_dst_memory, user_dst_memory));
            }
        }

        streams[0].submitAndWait();
        return ND4J_STATUS_OK;
    }
#endif
        nd4j_debug("MKL-DNN is not used for lrn!\n", 0);
        T* scaleBuffer = reinterpret_cast<T*>(scale->buffer());

        T tbias = static_cast<T>(bias);
        T tbeta = static_cast<T>(beta);
        T one(1.f);

        if (output->ews() == 1 && input->ews() == 1 && input->ordering() == 'c' && output->ordering() == 'c') {

            PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(2)
            for (int c = 0; c < chunkCount; c++) {
                for (int e = 0; e < lastDim; e++) {
                    int begin = nd4j::math::nd4j_max<int>(0, e - depth);
                    int end = nd4j::math::nd4j_min<int>(depth + e + 1, lastDim);
                    T quadSum = 0.f;
                    int shift = c * lastDim;
                    auto iX = inputBuffer + shift;

                    for (int pos = begin; pos < end; ++pos) {
                        T val = iX[pos]; //listInput->at(c)->t<T>(pos);
                        quadSum += val * val;
                    }
                    T aSum = alpha * quadSum;
                    T tXe = iX[e];
                    //scaleBuffer[shift + e] = (2. * alpha * tbeta) * tXe * tXe / math::nd4j_pow<T,T,T>(tbias + aSum, 1 + beta);
                    T dividor = nd4j::math::nd4j_pow<T, T, T>(tbias + aSum, tbeta);
                    outputBuffer[shift + e] = tXe / dividor;
                    scaleBuffer[shift + e] = outputBuffer[shift + e] * outputBuffer[shift + e] * (2. * alpha * tbeta) *  math::nd4j_pow<T,T,T>(tbias + aSum, tbeta - 1);
                }
            }
        } else {
            PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
            for (int c = 0; c < chunkCount; c++) {
                for (int e = 0; e < lastDim; e++) {
                    int begin = nd4j::math::nd4j_max(0, e - depth);
                    int end = nd4j::math::nd4j_min(depth + e + 1, lastDim);
                    T quadSum = 0;
                    int shift = c * lastDim;

                    PRAGMA_OMP_SIMD_SUM(quadSum)
                    for (int pos = begin; pos < end; ++pos) {
                        T val = inputBuffer[shape::getIndexOffset(shift + pos, input->getShapeInfo(), totalLength)]; //listInput->at(c)->t<T>(pos);
                        quadSum += val * val;
                    }

                    auto p = shape::getIndexOffset(shift + e, input->getShapeInfo(), totalLength);
                    T dividor = nd4j::math::nd4j_pow<T, T, T>(bias + alpha * quadSum, beta);
                    scaleBuffer[shift + e] = one - (alpha * inputBuffer[p] * inputBuffer[p] * 2 * beta) / dividor;
                    outputBuffer[shape::getIndexOffset(shift + e, output->shapeInfo(), totalLength)] = inputBuffer[p] / dividor;
                }
            }
        }

        return Status::OK();
    }

    BUILD_SINGLE_TEMPLATE(template int lrnFunctor_, (nd4j::graph::Context& block, NDArray* input, NDArray* output, int depth, float bias, float alpha, float beta), FLOAT_TYPES);

    int lrnFunctor(nd4j::graph::Context& block, NDArray* input, NDArray* output, int depth, double bias, double alpha, double beta) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return lrnFunctor_, (block, input, output, depth, bias, alpha, beta), FLOAT_TYPES);
    }

    int lrnFunctorEx(nd4j::graph::Context& block, NDArray* input, NDArray* output, NDArray* scale, int depth, double bias, double alpha, double beta) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return lrnFunctorEx_, (block, input, output, scale, depth, bias, alpha, beta), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template int lrnFunctorEx_, (nd4j::graph::Context& block, NDArray* input, NDArray* output, NDArray* scale, int depth, float bias, float alpha, float beta);, FLOAT_TYPES);

    int lrnFunctorEx(nd4j::graph::Context& block, NDArray* input, NDArray* output, NDArray* unitScale, NDArray* scale, int depth, double bias, double alpha, double beta) {
    
        depth = nd4j::math::nd4j_min<Nd4jLong>(depth, input->sizeAt(1));

        int halfDepth = depth / 2;
        halfDepth = nd4j::math::nd4j_max(halfDepth, 0);
        const int channel =  input->sizeAt(1);

#ifdef HAVE_MKLDNN
    if (block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported({input, scale, output})) {
        std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
        if (streams.empty()) {
            streams.push_back(MKLDNNStream("lrn_bp"));
        }

        if (streams[0].checkAndReset({input, scale}, {output}, {(float)bias, (float)alpha, (float)beta}, {depth})) {
            mkldnn_memory_desc_t empty;
            mkldnn::memory::desc lrn_src_md(empty), lrn_diff_src_md(empty), lrn_dst_md(empty), user_src_md(empty), user_diff_src_md(empty), user_dst_md(empty);

            getMKLDNNMemoryDescLrn(input, scale, output, &lrn_src_md, &lrn_diff_src_md, &lrn_dst_md, &user_src_md, &user_diff_src_md, &user_dst_md, 1);

            auto lrn_desc = lrn_forward::desc(prop_kind::forward, lrn_across_channels, lrn_src_md, (2 * halfDepth + 1), alpha * (2 * halfDepth + 1), beta, bias);
            auto lrn_back_desc = lrn_backward::desc(lrn_across_channels, lrn_src_md, lrn_diff_src_md, (2 * halfDepth + 1), alpha * (2 * halfDepth + 1), beta, bias);

            auto engine = streams[0].getEngine();
            auto lrn_prim_desc = lrn_forward::primitive_desc(lrn_desc, engine);
            auto lrn_back_prim_desc = lrn_backward::primitive_desc(lrn_back_desc, engine, lrn_prim_desc);
            auto user_src_memory = mkldnn::memory({user_src_md, engine}, input->buffer());
            auto user_dst_memory = mkldnn::memory({user_diff_src_md, engine}, scale->buffer());
            auto user_diff_src_memory = mkldnn::memory({user_dst_md, engine}, output->buffer());

            auto lrn_src_memory = user_src_memory;
            streams[0].addMemory(user_src_memory);
            if (mkldnn::memory::primitive_desc(lrn_prim_desc.src_primitive_desc())
                    != user_src_memory.get_primitive_desc()) {
                lrn_src_memory = mkldnn::memory(lrn_prim_desc.src_primitive_desc());
                streams[0].addMemory(lrn_src_memory);
                streams[0].addOperation(reorder(user_src_memory, lrn_src_memory));
            }

            auto lrn_diff_src_memory = user_diff_src_memory;
            streams[0].addMemory(user_diff_src_memory);
            if (mkldnn::memory::primitive_desc(lrn_back_prim_desc.diff_src_primitive_desc())
                    != user_diff_src_memory.get_primitive_desc()) {
                lrn_diff_src_memory = mkldnn::memory(lrn_back_prim_desc.diff_src_primitive_desc());
                streams[0].addMemory(lrn_diff_src_memory);
            }

            auto lrn_dst_memory = user_dst_memory;
            streams[0].addMemory(user_dst_memory);
            if (mkldnn::memory::primitive_desc(lrn_back_prim_desc.diff_dst_primitive_desc())
                    != user_dst_memory.get_primitive_desc()) {
                lrn_dst_memory = mkldnn::memory(lrn_back_prim_desc.diff_dst_primitive_desc());
                streams[0].addMemory(lrn_dst_memory);
                streams[0].addOperation(reorder(user_dst_memory, lrn_dst_memory));
            }

            streams[0].addOperation(lrn_backward(lrn_back_prim_desc, lrn_src_memory, lrn_dst_memory, lrn_diff_src_memory));

            if (mkldnn::memory::primitive_desc(lrn_back_prim_desc.diff_src_primitive_desc())
                    != user_diff_src_memory.get_primitive_desc()) {
                streams[0].addOperation(reorder(lrn_diff_src_memory, user_diff_src_memory));
            }
        }

        streams[0].submitAndWait();
        return ND4J_STATUS_OK;
    }
#endif
    nd4j_debug("MKL-DNN is not used for lrn_bp!\n", 0);

        std::unique_ptr<NDArray> activitySqr(input->dup('c'));//NDArrayFactory<T>::createUninitialized(input));
        std::unique_ptr<NDArray> sumPart(activitySqr->dup('c'));

        input->applyPairwiseTransform(pairwise::Multiply, input, activitySqr.get(), nullptr);

        PRAGMA_OMP_PARALLEL_FOR_IF(halfDepth + 1 > Environment::getInstance()->tadThreshold())
        for (int i = 1; i < halfDepth + 1; i++) {
            std::vector<Nd4jLong> indA = {0,0, i,channel, 0,0, 0,0};
            std::vector<Nd4jLong> indB = {0,0, 0,channel-i, 0,0, 0,0};

            NDArray tmp = (*sumPart)(indA, true);
            NDArray addVal = (*activitySqr)(indB, true);

            tmp.applyPairwiseTransform(pairwise::Add, addVal, nullptr);


            NDArray tmp2 = (*sumPart)(indB, true);
            NDArray addVal2 = (*activitySqr)(indA, true);

            tmp2.applyPairwiseTransform(pairwise::Add, addVal2, nullptr);
        }

        if (unitScale != nullptr && scale != nullptr) {
            sumPart->applyScalar(scalar::Multiply, alpha, unitScale, nullptr);
            unitScale->applyScalar(scalar::Add, bias);

            float p = static_cast<float>(-beta);
            unitScale->applyScalar(scalar::Pow, p, scale, nullptr);
            input->applyPairwiseTransform(pairwise::Multiply, scale, output, nullptr);
        }

        return Status::OK();
    }

}
}
}
