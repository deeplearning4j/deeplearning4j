/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/helpers/lrn.h>
#include <Status.h>
#include <ConstantTadHelper.h>

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

    const int rank = input->rankOf();

    TadPack inTadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), {rank - 1});
    TadPack outTadPack;

    if(shape::haveSameShapeAndStrides(input->getShapeInfo(), output->getShapeInfo()))
        outTadPack = inTadPack;
    else
        outTadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), {rank - 1});

    const Nd4jLong numOfTads = inTadPack.numberOfTads();
    const Nd4jLong tadLen    = input->sizeAt(-1); 
    
    const Nd4jLong* inTadOffsets    = inTadPack.primaryOffsets();        
    const Nd4jLong* outTadOffsets = outTadPack.primaryOffsets();

    const Nd4jLong inTadEws    = shape::elementWiseStride(inTadPack.primaryShapeInfo());
    const Nd4jLong outTadEws = shape::elementWiseStride(outTadPack.primaryShapeInfo());
    
    const T* inBuff  = reinterpret_cast<T*>(input->getBuffer());
          T* outBuff = reinterpret_cast<T*>(output->getBuffer());

    const T tbias  = static_cast<T>(bias);
    const T tbeta  = static_cast<T>(beta);
    const T talpha = static_cast<T>(alpha);    

    if(inTadEws == 1 && outTadEws == 1) {
        
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (uint i = 0; i < numOfTads; ++i) {
            const T* x = inBuff    + inTadOffsets[i];
                  T* y = outBuff + outTadOffsets[i];

            T prev = 0;

            // calculate squared sum of elements per each j-th element range [j - depth, j + depth + 1]
            // we store each squared sum in corresponding element of y array
            for (uint j = 0; j < tadLen; ++j) {
                const uint begin = nd4j::math::nd4j_max<int>(0, j - depth);
                const uint last  = depth + j + 1;           
                const uint end   = nd4j::math::nd4j_min<int>(last, tadLen);
                
                if (j == 0) {                    
                    for (uint s = begin; s < end; ++s)
                        prev = prev + x[s] * x[s];
                    y[j] = prev;
                }
                else if (begin == 0 && last <= tadLen)
                    y[j] = prev + x[end - 1] * x[end - 1];
                else if (begin > 0 && last <= tadLen)
                    y[j] = prev + x[end - 1] * x[end - 1] - x[begin - 1] * x[begin - 1];
                else if (begin > 0 && last > tadLen)
                    y[j] = prev - x[begin - 1] * x[begin - 1];
                else
                    y[j] = prev;

                  if(j != 0)
                    prev = y[j];
                
                y[j] = x[j] / nd4j::math::nd4j_pow<T, T, T>(tbias + alpha * prev, tbeta); 
            }          
        }
    }
    else {
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (uint i = 0; i < numOfTads; ++i) {
            const T* x = inBuff    + inTadOffsets[i];
                  T* y = outBuff + outTadOffsets[i];

            T prev = 0;

            // calculate squared sum of elements per each j-th element range [j - depth, j + depth + 1]
            // we store each squared sum in corresponding element of y array
            for (uint j = 0; j < tadLen; ++j) {
                const uint begin = nd4j::math::nd4j_max<int>(0, j - depth);
                const uint last  = depth + j + 1;           
                const uint end   = nd4j::math::nd4j_min<int>(last, tadLen);
                
                if (j == 0) {                    
                    for (uint s = begin; s < end; ++s)
                        prev = prev + x[s*inTadEws] * x[s*inTadEws];
                    y[j*outTadEws] = prev;
                }
                else if (begin == 0 && last <= tadLen)
                    y[j*outTadEws] = prev + x[(end - 1)*inTadEws] * x[(end - 1)*inTadEws];
                else if (begin > 0 && last <= tadLen)
                    y[j*outTadEws] = prev + x[(end - 1)*inTadEws] * x[(end - 1)*inTadEws] - x[(begin - 1)*inTadEws] * x[(begin - 1)*inTadEws];
                else if (begin > 0 && last > tadLen)
                    y[j*outTadEws] = prev - x[(begin - 1)*inTadEws] * x[(begin - 1)*inTadEws];
                else
                    y[j*outTadEws] = prev;

                  if(j != 0)
                    prev = y[j*outTadEws];
                
                y[j*outTadEws] = x[j*inTadEws] / nd4j::math::nd4j_pow<T, T, T>(tbias + alpha * prev, tbeta); 
            }          
        }
    }    
    return Status::OK();
}
    
BUILD_SINGLE_TEMPLATE(template int lrnFunctor_, (nd4j::graph::Context& block, NDArray* input, NDArray* output, int depth, float bias, float alpha, float beta), FLOAT_TYPES);

int lrnFunctor(nd4j::graph::Context& block, NDArray* input, NDArray* output, int depth, double bias, double alpha, double beta) {
    BUILD_SINGLE_SELECTOR(input->dataType(), return lrnFunctor_, (block, input, output, depth, bias, alpha, beta), FLOAT_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void lrnBP_(const NDArray& input, const NDArray& gradO, NDArray& gradI, const int depth, const float bias, const float alpha, const float beta) {
    
    const int rank = input.rankOf();

    TadPack inTadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input.getShapeInfo(), {rank - 1});
    TadPack gradITadPack;

    if(shape::haveSameShapeAndStrides(input.getShapeInfo(), gradI.getShapeInfo()))
        gradITadPack = inTadPack;
    else
        gradITadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(gradI.getShapeInfo(), {rank - 1});

    const Nd4jLong numOfTads = inTadPack.numberOfTads();
    const Nd4jLong tadLen    = input.sizeAt(-1); 
    
    const Nd4jLong* inTadOffsets    = inTadPack.primaryOffsets();        
    const Nd4jLong* gradITadOffsets = gradITadPack.primaryOffsets();

    const Nd4jLong inTadEws    = shape::elementWiseStride(inTadPack.primaryShapeInfo());
    const Nd4jLong gradITadEws = shape::elementWiseStride(gradITadPack.primaryShapeInfo());
    
    const X* inBuff    = reinterpret_cast<X*>(input.getBuffer());
          Y* gradIBuff = reinterpret_cast<Y*>(gradI.getBuffer());    

    const Y tbias  = static_cast<Y>(bias);
    const Y tbeta  = static_cast<Y>(beta);
    const Y talpha = static_cast<Y>(alpha);
    const Y coeff  = talpha * tbeta; 

    if(inTadEws == 1 && gradITadEws == 1) {
        
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (uint i = 0; i < numOfTads; ++i) {
            const X* x = inBuff    + inTadOffsets[i];
                  Y* y = gradIBuff + gradITadOffsets[i];

            // this loop calculates squared sum of elements per each j-th element range [j - depth, j + depth + 1]
            // we store each squared sum in corresponding element of y array
            for (uint j = 0; j < tadLen; ++j) {
                const uint begin = nd4j::math::nd4j_max<int>(0, j - depth);
                const uint last  = depth + j + 1;           
                const uint end   = nd4j::math::nd4j_min<int>(last, tadLen);
                
                if (j == 0) {
                    y[0] = 0;
                    for (uint s = begin; s < end; ++s)
                        y[0] = y[0] + x[s] * x[s];
                }
                else if (begin == 0 && last <= tadLen)
                    y[j] = y[j - 1] + x[end - 1] * x[end - 1];
                else if (begin > 0 && last <= tadLen)
                    y[j] = y[j - 1] + x[end - 1] * x[end - 1] - x[begin - 1] * x[begin - 1];
                else if (begin > 0 && last > tadLen)
                    y[j] = y[j - 1] - x[begin - 1] * x[begin - 1];
                else
                    y[j] = y[j - 1];                
            }

            Y* factor = new Y[tadLen];

            Y prev = 0;
            // second loop calculates derivatives using information gained in first loop above
            for (uint j = 0; j < tadLen; ++j) {
                const uint begin = nd4j::math::nd4j_max<int>(0, j - depth);
                const uint last  = depth + j + 1;
                const uint end   = nd4j::math::nd4j_min<int>(last, tadLen);

                Y init = tbias + talpha * y[j];

                if (j == 0) {                    
                    for (uint s = begin; s < end; ++s) {
                        factor[s] = nd4j::math::nd4j_pow<Y, Y, Y>(tbias + talpha * y[s], -tbeta - 1);
                        prev = prev + x[s] * factor[s];
                    }
                    y[0] = prev;
                }
                else if(begin == 0 && last <= tadLen) {
                    factor[end - 1] = nd4j::math::nd4j_pow<Y, Y, Y>(tbias + talpha * y[end - 1], -tbeta - 1);
                    y[j] = prev + x[end - 1] * factor[end - 1];
                }
                else if (begin > 0 && last <= tadLen) {
                    factor[end - 1] = nd4j::math::nd4j_pow<Y, Y, Y>(tbias + talpha * y[end - 1], -tbeta - 1);
                    y[j] = prev + x[end - 1] * factor[end - 1] - x[begin - 1] * factor[begin - 1];
                }
                else if (begin > 0 && last > tadLen)
                    y[j] = prev - x[begin - 1] * factor[begin - 1];
                else 
                    y[j] = prev;
                
                if(j != 0)
                    prev = y[j];

                y[j] = factor[j] * init - 2 * x[j] * coeff * prev;                
            }
            
            delete []factor;
        }
    }
    else {

        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (uint i = 0; i < numOfTads; ++i) {
            const X* x = inBuff    + inTadOffsets[i];
                  Y* y = gradIBuff + gradITadOffsets[i];

            // this loop calculates squared sum of elements per each j-th element range [j - depth, j + depth + 1]
            // we store each squared sum in corresponding element of y array
            for (uint j = 0; j < tadLen; ++j) {
                const uint begin = nd4j::math::nd4j_max<int>(0, j - depth);
                const uint last  = depth + j + 1;           
                const uint end   = nd4j::math::nd4j_min<int>(last, tadLen);
                
                if (j == 0) {
                    y[0] = 0;
                    for (uint s = begin; s < end; ++s)
                        y[0] = y[0] + x[s*inTadEws] * x[s*inTadEws];
                }
                else if (begin == 0 && last <= tadLen)
                    y[j*gradITadEws] = y[(j - 1)*gradITadEws] + x[(end - 1)*inTadEws] * x[(end - 1)*inTadEws];
                else if (begin > 0 && last <= tadLen)
                    y[j*gradITadEws] = y[(j - 1)*gradITadEws] + x[(end - 1)*inTadEws] * x[(end - 1)*inTadEws] - x[(begin - 1)*inTadEws] * x[(begin - 1)*inTadEws];
                else if (begin > 0 && last > tadLen)
                    y[j*gradITadEws] = y[(j - 1)*gradITadEws] - x[(begin - 1)*inTadEws] * x[(begin - 1)*inTadEws];
                else
                    y[j*gradITadEws] = y[(j - 1)*gradITadEws];              
            }

            Y* factor = new Y[tadLen];

            Y prev = 0;
            // second loop calculates derivatives using information gained in first loop above
            for (uint j = 0; j < tadLen; ++j) {
                const uint begin = nd4j::math::nd4j_max<int>(0, j - depth);
                const uint last  = depth + j + 1;
                const uint end   = nd4j::math::nd4j_min<int>(last, tadLen);

                Y init = tbias + talpha * y[j*gradITadEws];

                if (j == 0) {                    
                    for (uint s = begin; s < end; ++s) {
                        factor[s] = nd4j::math::nd4j_pow<Y, Y, Y>(tbias + talpha * y[s*gradITadEws], -tbeta - 1);
                        prev = prev + x[s*inTadEws] * factor[s];
                    }
                    y[0] = prev;
                }
                else if(begin == 0 && last <= tadLen) {
                    factor[end - 1] = nd4j::math::nd4j_pow<Y, Y, Y>(tbias + talpha * y[(end - 1)*gradITadEws], -tbeta - 1);
                    y[j*gradITadEws] = prev + x[(end - 1)*inTadEws] * factor[end - 1];
                }
                else if (begin > 0 && last <= tadLen) {
                    factor[end - 1] = nd4j::math::nd4j_pow<Y, Y, Y>(tbias + talpha * y[(end - 1)*gradITadEws], -tbeta - 1);
                    y[j*gradITadEws] = prev + x[(end - 1)*inTadEws] * factor[end - 1] - x[(begin - 1)*inTadEws] * factor[begin - 1];
                }
                else if (begin > 0 && last > tadLen)
                    y[j*gradITadEws] = prev - x[(begin - 1)*inTadEws] * factor[begin - 1];
                else 
                    y[j*gradITadEws] = prev;
                
                if(j != 0)
                    prev = y[j*gradITadEws];

                y[j*gradITadEws] = factor[j] * init - 2 * x[j*inTadEws] * coeff * prev;                
            }
            
            delete []factor;
        }
    }    
    gradI *= gradO;
}

BUILD_DOUBLE_TEMPLATE(template void lrnBP_, (const NDArray& input, const NDArray& gradO, NDArray& gradI, const int depth, const float bias, const float alpha, const float beta), LIBND4J_TYPES, FLOAT_TYPES);

void lrnBP(const NDArray& input, const NDArray& gradO, NDArray& gradI, const int depth, const float bias, const float alpha, const float beta) {
    BUILD_DOUBLE_SELECTOR(input.dataType(), gradO.dataType(), lrnBP_, (input, gradO, gradI, depth, bias, alpha, beta), LIBND4J_TYPES, FLOAT_TYPES);
}

}
}
}
