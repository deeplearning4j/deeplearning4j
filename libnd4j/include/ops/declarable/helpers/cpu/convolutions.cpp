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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 18.09.2018
//

#include<ops/declarable/helpers/addBias.h>
#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/im2col.h>
#include <ops/declarable/helpers/col2im.h>
#include <NDArrayFactory.h>
#include <MmulHelper.h>

namespace nd4j {
namespace ops  {

//////////////////////////////////////////////////////////////////////////
// [bS, iC, iD, iH, iW] is convoluted to [bS, iC, kD, kH, kW, oD, oH, oW]        
template <typename T>
static void vol2col_(const NDArray& volume, NDArray& columns, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    const int bS = volume.sizeAt(0);
    const int iC = volume.sizeAt(1);
    const int iD = volume.sizeAt(2);
    const int iH = volume.sizeAt(3);
    const int iW = volume.sizeAt(4);
    const int kD = columns.sizeAt(2);
    const int kH = columns.sizeAt(3);
    const int kW = columns.sizeAt(4);
    const int oD = columns.sizeAt(5);
    const int oH = columns.sizeAt(6);
    const int oW = columns.sizeAt(7);
    const Nd4jLong colStride0 = columns.stridesOf()[0];
    const Nd4jLong colStride1 = columns.stridesOf()[1];
    const Nd4jLong colStride2 = columns.stridesOf()[2];
    const Nd4jLong colStride3 = columns.stridesOf()[3];
    const Nd4jLong colStride4 = columns.stridesOf()[4];
    const Nd4jLong colStride5 = columns.stridesOf()[5];
    const Nd4jLong colStride6 = columns.stridesOf()[6];
    const Nd4jLong colStride7 = columns.stridesOf()[7];  
    const Nd4jLong volStride0 = volume.stridesOf()[0];
    const Nd4jLong volStride1 = volume.stridesOf()[1];
    const Nd4jLong volStride2 = volume.stridesOf()[2];
    const Nd4jLong volStride3 = volume.stridesOf()[3];
    const Nd4jLong volStride4 = volume.stridesOf()[4];
    
    T* colBuff = columns.bufferAsT<T>();
    T* volBuff = const_cast<NDArray&>(volume).bufferAsT<T>();

    T *col, *vol;
    int volDep, volRow, volCol;

if (volume.ordering() == 'c' &&  columns.ordering() == 'c' && shape::strideDescendingCAscendingF(volume.getShapeInfo()) && shape::strideDescendingCAscendingF(columns.getShapeInfo()))

    PRAGMA_OMP_PARALLEL_FOR_ARGS(private(col, vol, volDep, volRow, volCol) collapse(2))
    for (int b = 0; b < bS; b++) {
        for (int c = 0; c < iC; ++c) {        
            for (int kDep = 0; kDep < kD; ++kDep) { 
                for (int kRow = 0; kRow < kH; ++kRow) {                        
                    for (int kCol = 0; kCol < kW; ++kCol) {                            
                        for (int colD = 0; colD < oD; ++colD) {
                            for (int colH = 0; colH < oH; ++colH) {
                                for (int colW = 0; colW < oW; ++colW) {                    
                                
                                    volDep = (-pD + kDep * dD) + colD*sD;
                                    volRow = (-pH + kRow * dH) + colH*sH;
                                    volCol = (-pW + kCol * dW) + colW*sW;
                                        
                                    col = colBuff + b*colStride0 + c*colStride1 + kDep*colStride2 + kRow*colStride3 + kCol*colStride4 + colD*colStride5 + colH*colStride6 + colW*colStride7;
                                    vol = volBuff + b*volStride0 + c*volStride1 + volDep*volStride2 + volRow*volStride3 + volCol*volStride4;
                                                    
                                    if (static_cast<unsigned>(volDep) >= static_cast<unsigned>(iD) || static_cast<unsigned>(volRow) >= static_cast<unsigned>(iH) || static_cast<unsigned>(volCol) >= static_cast<unsigned>(iW))
                                        *col = static_cast<T>(0.);
                                    else 
                                        *col = *vol;
                                }
                            }
                        }
                    }
                }
            }
        }
    }  

else

    PRAGMA_OMP_PARALLEL_FOR_ARGS(private(vol, col, volDep, volRow, volCol))
    for (int b = 0; b < bS; b++) {
        for (int colD = 0; colD < oD; ++colD) {
            for (int colH = 0; colH < oH; ++colH) {
                for (int colW = 0; colW < oW; ++colW) {
                    for (int c = 0; c < iC; ++c) {
                        for (int kDep = 0; kDep < kD; ++kDep) { 
                            for (int kRow = 0; kRow < kH; ++kRow) {                        
                                for (int kCol = 0; kCol < kW; ++kCol) {                            
                        
                                    volDep = (-pD + kDep * dD) + colD*sD;
                                    volRow = (-pH + kRow * dH) + colH*sH;
                                    volCol = (-pW + kCol * dW) + colW*sW;
                                        
                                    col = colBuff + b*colStride0 + c*colStride1 + kDep*colStride2 + kRow*colStride3 + kCol*colStride4 + colD*colStride5 + colH*colStride6 + colW*colStride7;
                                    vol = volBuff + b*volStride0 + c*volStride1 + volDep*volStride2 + volRow*volStride3 + volCol*volStride4;
                                                    
                                    if (static_cast<unsigned>(volDep) >= static_cast<unsigned>(iD) || static_cast<unsigned>(volRow) >= static_cast<unsigned>(iH) || static_cast<unsigned>(volCol) >= static_cast<unsigned>(iW))
                                        *col = static_cast<T>(0.);
                                    else 
                                        *col = *vol;
                                }
                            }
                        }
                    }
                }
            }
        }
    }  
}

//////////////////////////////////////////////////////////////////////////
// [bS, iC, kD, kH, kW, oD, oH, oW] is de-convoluted to [bS, iC, iD, iH, iW]
template <typename T>
static void col2vol_(const NDArray& columns, NDArray& volume, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    const int bS = volume.sizeAt(0);
    const int iC = volume.sizeAt(1);
    const int iD = volume.sizeAt(2);
    const int iH = volume.sizeAt(3);
    const int iW = volume.sizeAt(4);
    const int kD = columns.sizeAt(2);
    const int kH = columns.sizeAt(3);
    const int kW = columns.sizeAt(4);
    const int oD = columns.sizeAt(5);
    const int oH = columns.sizeAt(6);
    const int oW = columns.sizeAt(7);
    const Nd4jLong colStride0 = columns.stridesOf()[0];
    const Nd4jLong colStride1 = columns.stridesOf()[1];
    const Nd4jLong colStride2 = columns.stridesOf()[2];
    const Nd4jLong colStride3 = columns.stridesOf()[3];
    const Nd4jLong colStride4 = columns.stridesOf()[4];
    const Nd4jLong colStride5 = columns.stridesOf()[5];
    const Nd4jLong colStride6 = columns.stridesOf()[6];
    const Nd4jLong colStride7 = columns.stridesOf()[7];  
    const Nd4jLong volStride0 = volume.stridesOf()[0];
    const Nd4jLong volStride1 = volume.stridesOf()[1];
    const Nd4jLong volStride2 = volume.stridesOf()[2];
    const Nd4jLong volStride3 = volume.stridesOf()[3];
    const Nd4jLong volStride4 = volume.stridesOf()[4];    
    
    T* volBuff = volume.bufferAsT<T>();
    T* colBuff = const_cast<NDArray&>(columns).bufferAsT<T>();

    // initial zeroing of volume content
    memset(volBuff, 0, volume.lengthOf() * sizeof(T));

    T* col, *vol;
    int volDep, volRow, volCol;

if (volume.ordering() == 'c' &&  columns.ordering() == 'c' && shape::strideDescendingCAscendingF(volume.getShapeInfo()) && shape::strideDescendingCAscendingF(columns.getShapeInfo()))

    PRAGMA_OMP_PARALLEL_FOR_ARGS(private(col, vol, volDep, volRow, volCol) collapse(2))
    for (int b = 0; b < bS; b++) {        
        for (int c = 0; c < iC; ++c) {        
            for (int kDep = 0; kDep < kD; ++kDep) { 
                for (int kRow = 0; kRow < kH; ++kRow) {                        
                    for (int kCol = 0; kCol < kW; ++kCol) {                            
                        for (int colD = 0; colD < oD; ++colD) {
                            for (int colH = 0; colH < oH; ++colH) {
                                for (int colW = 0; colW < oW; ++colW) {                    

                                    volDep = (-pD + kDep * dD) + colD*sD;
                                    volRow = (-pH + kRow * dH) + colH*sH;
                                    volCol = (-pW + kCol * dW) + colW*sW;

                                    col = colBuff + b*colStride0 + c*colStride1 + kDep*colStride2 + kRow*colStride3 + kCol*colStride4 + colD*colStride5 + colH*colStride6 + colW*colStride7;
                                    vol = volBuff + b*volStride0 + c*volStride1 + volDep*volStride2 + volRow*volStride3 + volCol*volStride4;

                                    if (static_cast<unsigned>(volDep) < static_cast<unsigned>(iD) && static_cast<unsigned>(volRow) < static_cast<unsigned>(iH) && static_cast<unsigned>(volCol) < static_cast<unsigned>(iW))
                                        *vol += *col;
                                }
                            }
                        }
                    }
                }
            }
        }
    }  

else

    PRAGMA_OMP_PARALLEL_FOR_ARGS(private(vol, col, volDep, volRow, volCol))
    for (int b = 0; b < bS; b++) {
        for (int colD = 0; colD < oD; ++colD) {
            for (int colH = 0; colH < oH; ++colH) {
                for (int colW = 0; colW < oW; ++colW) {
                    for (int c = 0; c < iC; ++c) {
                        for (int kDep = 0; kDep < kD; ++kDep) { 
                            for (int kRow = 0; kRow < kH; ++kRow) {                        
                                for (int kCol = 0; kCol < kW; ++kCol) {                            
                        
                                    volDep = (-pD + kDep * dD) + colD*sD;
                                    volRow = (-pH + kRow * dH) + colH*sH;
                                    volCol = (-pW + kCol * dW) + colW*sW;
                                        
                                    col = colBuff + b*colStride0 + c*colStride1 + kDep*colStride2 + kRow*colStride3 + kCol*colStride4 + colD*colStride5 + colH*colStride6 + colW*colStride7;
                                    vol = volBuff + b*volStride0 + c*volStride1 + volDep*volStride2 + volRow*volStride3 + volCol*volStride4;
                                                    
                                    if (static_cast<unsigned>(volDep) < static_cast<unsigned>(iD) && static_cast<unsigned>(volRow) < static_cast<unsigned>(iH) && static_cast<unsigned>(volCol) < static_cast<unsigned>(iW))
                                        *vol += *col;
                                }
                            }
                        }
                    }
                }
            }
        }
    }  
}


#ifdef HAVE_MKLDNN
using namespace mkldnn;

void ConvolutionUtils::getMKLDNNMemoryDescConv2d(
        int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, bool isNCHW,
        int bS, int iC, int iH, int iW, int oC, int oH, int oW, const NDArray* src, const NDArray* diff_src,
        const NDArray* weights, const NDArray* diff_weights, const NDArray* bias, const NDArray* dst,
        mkldnn::memory::desc* conv_src_md, mkldnn::memory::desc* conv_diff_src_md, mkldnn::memory::desc* conv_weights_md,
        mkldnn::memory::desc* conv_diff_weights_md, mkldnn::memory::desc* conv_bias_md, mkldnn::memory::desc* conv_dst_md,
        mkldnn::memory::desc* user_src_md, mkldnn::memory::desc* user_diff_src_md, mkldnn::memory::desc* user_weights_md,
        mkldnn::memory::desc* user_diff_weights_md, mkldnn::memory::desc* user_bias_md, mkldnn::memory::desc* user_dst_md,
        mkldnn::memory::dims& conv_strides, mkldnn::memory::dims& conv_padding, mkldnn::memory::dims& conv_padding_r) {
    mkldnn::memory::dims conv_src_tz = { bS, iC, iH, iW };
    mkldnn::memory::dims conv_weights_tz = { oC, iC, kH, kW };
    mkldnn::memory::dims conv_bias_tz = { oC };
    mkldnn::memory::dims conv_dst_tz = { bS, oC, oH, oW };

    conv_strides = { sH, sW };
    conv_padding = { pH, pW };
    conv_padding_r = { (oH - 1) * sH - iH + kH - pH,
                       (oW - 1) * sW - iW + kW - pW };

    auto type = mkldnn::memory::data_type::f32;
    auto format = isNCHW ? mkldnn::memory::format::nchw : mkldnn::memory::format::nhwc;
    auto formatw = mkldnn::memory::format::hwio;

    if (src != nullptr && conv_src_md != nullptr) {
        *conv_src_md = mkldnn::memory::desc({ conv_src_tz }, type, mkldnn::memory::format::any);
        *user_src_md = mkldnn::memory::desc({ conv_src_tz }, type, format);
        user_src_md->data.format = mkldnn_blocked; // overrides "format = isNCHW ? nchw : nhwc"
        user_src_md->data.layout_desc.blocking.strides[0][0] = src->stridesOf()[isNCHW ? 0 : 0];
        user_src_md->data.layout_desc.blocking.strides[0][1] = src->stridesOf()[isNCHW ? 1 : 3];
        user_src_md->data.layout_desc.blocking.strides[0][2] = src->stridesOf()[isNCHW ? 2 : 1];
        user_src_md->data.layout_desc.blocking.strides[0][3] = src->stridesOf()[isNCHW ? 3 : 2];
    }

    if (diff_src != nullptr && conv_diff_src_md != nullptr) {
        *conv_diff_src_md = mkldnn::memory::desc({ conv_src_tz }, type, mkldnn::memory::format::any);
        *user_diff_src_md = mkldnn::memory::desc({ conv_src_tz }, type, format);
        user_diff_src_md->data.format = mkldnn_blocked; // overrides "format = isNCHW ? nchw : nhwc"
        user_diff_src_md->data.layout_desc.blocking.strides[0][0] = diff_src->stridesOf()[isNCHW ? 0 : 0];
        user_diff_src_md->data.layout_desc.blocking.strides[0][1] = diff_src->stridesOf()[isNCHW ? 1 : 3];
        user_diff_src_md->data.layout_desc.blocking.strides[0][2] = diff_src->stridesOf()[isNCHW ? 2 : 1];
        user_diff_src_md->data.layout_desc.blocking.strides[0][3] = diff_src->stridesOf()[isNCHW ? 3 : 2];
    }

    if (weights != nullptr && conv_weights_md != nullptr) {
        *conv_weights_md = mkldnn::memory::desc({ conv_weights_tz }, type, mkldnn::memory::format::any);
        *user_weights_md = mkldnn::memory::desc({ conv_weights_tz }, type, formatw);
        user_weights_md->data.format = mkldnn_blocked; // overrides "formatw = hwio"
        user_weights_md->data.layout_desc.blocking.strides[0][0] = weights->stridesOf()[3];
        user_weights_md->data.layout_desc.blocking.strides[0][1] = weights->stridesOf()[2];
        user_weights_md->data.layout_desc.blocking.strides[0][2] = weights->stridesOf()[0];
        user_weights_md->data.layout_desc.blocking.strides[0][3] = weights->stridesOf()[1];
    }

    if (diff_weights != nullptr && conv_diff_weights_md != nullptr) {
        *conv_diff_weights_md = mkldnn::memory::desc({ conv_weights_tz }, type, mkldnn::memory::format::any);
        *user_diff_weights_md = mkldnn::memory::desc({ conv_weights_tz }, type, formatw);
        user_diff_weights_md->data.format = mkldnn_blocked; // overrides "formatw = hwio"
        user_diff_weights_md->data.layout_desc.blocking.strides[0][0] = diff_weights->stridesOf()[3];
        user_diff_weights_md->data.layout_desc.blocking.strides[0][1] = diff_weights->stridesOf()[2];
        user_diff_weights_md->data.layout_desc.blocking.strides[0][2] = diff_weights->stridesOf()[0];
        user_diff_weights_md->data.layout_desc.blocking.strides[0][3] = diff_weights->stridesOf()[1];
    }

    if (bias != nullptr && conv_bias_md != nullptr) {
        *conv_bias_md = mkldnn::memory::desc({ conv_bias_tz }, type, mkldnn::memory::format::any);
        *user_bias_md = mkldnn::memory::desc({ conv_bias_tz }, type, mkldnn::memory::format::x);
    }

    if (dst != nullptr && conv_dst_md != nullptr) {
        *conv_dst_md = mkldnn::memory::desc({ conv_dst_tz }, type, mkldnn::memory::format::any);
        *user_dst_md = mkldnn::memory::desc({ conv_dst_tz }, type, format);
        user_dst_md->data.format = mkldnn_blocked; // overrides "format = isNCHW ? nchw : nhwc"
        user_dst_md->data.layout_desc.blocking.strides[0][0] = dst->stridesOf()[isNCHW ? 0 : 0];
        user_dst_md->data.layout_desc.blocking.strides[0][1] = dst->stridesOf()[isNCHW ? 1 : 3];
        user_dst_md->data.layout_desc.blocking.strides[0][2] = dst->stridesOf()[isNCHW ? 2 : 1];
        user_dst_md->data.layout_desc.blocking.strides[0][3] = dst->stridesOf()[isNCHW ? 3 : 2];
    }
}

void ConvolutionUtils::getMKLDNNMemoryDescConv3d(
        int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW, int dD, int dH, int dW, bool isSameMode, bool isNCDHW,
        int bS, int iC, int iD, int iH, int iW, int oC, int oD, int oH, int oW, const NDArray* src, const NDArray* diff_src,
        const NDArray* weights, const NDArray* diff_weights, const NDArray* bias, const NDArray* dst,
        mkldnn::memory::desc* conv_src_md, mkldnn::memory::desc* conv_diff_src_md, mkldnn::memory::desc* conv_weights_md,
        mkldnn::memory::desc* conv_diff_weights_md, mkldnn::memory::desc* conv_bias_md, mkldnn::memory::desc* conv_dst_md,
        mkldnn::memory::desc* user_src_md, mkldnn::memory::desc* user_diff_src_md, mkldnn::memory::desc* user_weights_md,
        mkldnn::memory::desc* user_diff_weights_md, mkldnn::memory::desc* user_bias_md, mkldnn::memory::desc* user_dst_md,
        mkldnn::memory::dims& conv_strides, mkldnn::memory::dims& conv_padding, mkldnn::memory::dims& conv_padding_r) {
    mkldnn::memory::dims conv_src_tz = { bS, iC, iD, iH, iW };
    mkldnn::memory::dims conv_weights_tz = { oC, iC, kD, kH, kW };
    mkldnn::memory::dims conv_bias_tz = { oC };
    mkldnn::memory::dims conv_dst_tz = { bS, oC, oD, oH, oW };

    conv_strides = { sD, sH, sW };
    conv_padding = { pD, pH, pW };
    conv_padding_r = { (oD - 1) * sD - iD + kD - pD,
                       (oH - 1) * sH - iH + kH - pH,
                       (oW - 1) * sW - iW + kW - pW };

    auto type = mkldnn::memory::data_type::f32;
    auto format = isNCDHW ? mkldnn::memory::format::ncdhw : mkldnn::memory::format::ndhwc;
    auto formatw = mkldnn::memory::format::dhwio;

    if (src != nullptr && conv_src_md != nullptr) {
        *conv_src_md = mkldnn::memory::desc({ conv_src_tz }, type, mkldnn::memory::format::any);
        *user_src_md = mkldnn::memory::desc({ conv_src_tz }, type, format);
        user_src_md->data.format = mkldnn_blocked; // overrides "format = isNCDHW ? ncdhw : ndhwc"
        user_src_md->data.layout_desc.blocking.strides[0][0] = src->stridesOf()[isNCDHW ? 0 : 0];
        user_src_md->data.layout_desc.blocking.strides[0][1] = src->stridesOf()[isNCDHW ? 1 : 4];
        user_src_md->data.layout_desc.blocking.strides[0][2] = src->stridesOf()[isNCDHW ? 2 : 1];
        user_src_md->data.layout_desc.blocking.strides[0][3] = src->stridesOf()[isNCDHW ? 3 : 2];
        user_src_md->data.layout_desc.blocking.strides[0][4] = src->stridesOf()[isNCDHW ? 4 : 3];
    }

    if (diff_src != nullptr && conv_diff_src_md != nullptr) {
        *conv_diff_src_md = mkldnn::memory::desc({ conv_src_tz }, type, mkldnn::memory::format::any);
        *user_diff_src_md = mkldnn::memory::desc({ conv_src_tz }, type, format);
        user_diff_src_md->data.format = mkldnn_blocked; // overrides "format = isNCDHW ? ncdhw : ndhwc"
        user_diff_src_md->data.layout_desc.blocking.strides[0][0] = diff_src->stridesOf()[isNCDHW ? 0 : 0];
        user_diff_src_md->data.layout_desc.blocking.strides[0][1] = diff_src->stridesOf()[isNCDHW ? 1 : 4];
        user_diff_src_md->data.layout_desc.blocking.strides[0][2] = diff_src->stridesOf()[isNCDHW ? 2 : 1];
        user_diff_src_md->data.layout_desc.blocking.strides[0][3] = diff_src->stridesOf()[isNCDHW ? 3 : 2];
        user_diff_src_md->data.layout_desc.blocking.strides[0][4] = diff_src->stridesOf()[isNCDHW ? 4 : 3];
    }

    if (weights != nullptr && conv_weights_md != nullptr) {
        *conv_weights_md = mkldnn::memory::desc({ conv_weights_tz }, type, mkldnn::memory::format::any);
        *user_weights_md = mkldnn::memory::desc({ conv_weights_tz }, type, formatw);
        user_weights_md->data.format = mkldnn_blocked; // overrides "formatw = dhwio"
        user_weights_md->data.layout_desc.blocking.strides[0][0] = weights->stridesOf()[4];
        user_weights_md->data.layout_desc.blocking.strides[0][1] = weights->stridesOf()[3];
        user_weights_md->data.layout_desc.blocking.strides[0][2] = weights->stridesOf()[0];
        user_weights_md->data.layout_desc.blocking.strides[0][3] = weights->stridesOf()[1];
        user_weights_md->data.layout_desc.blocking.strides[0][4] = weights->stridesOf()[2];
    }

    if (diff_weights != nullptr && conv_diff_weights_md != nullptr) {
        *conv_diff_weights_md = mkldnn::memory::desc({ conv_weights_tz }, type, mkldnn::memory::format::any);
        *user_diff_weights_md = mkldnn::memory::desc({ conv_weights_tz }, type, formatw);
        user_diff_weights_md->data.format = mkldnn_blocked; // overrides "formatw = dhwio"
        user_diff_weights_md->data.layout_desc.blocking.strides[0][0] = diff_weights->stridesOf()[4];
        user_diff_weights_md->data.layout_desc.blocking.strides[0][1] = diff_weights->stridesOf()[3];
        user_diff_weights_md->data.layout_desc.blocking.strides[0][2] = diff_weights->stridesOf()[0];
        user_diff_weights_md->data.layout_desc.blocking.strides[0][3] = diff_weights->stridesOf()[1];
        user_diff_weights_md->data.layout_desc.blocking.strides[0][4] = diff_weights->stridesOf()[2];
    }

    if (bias != nullptr && conv_bias_md != nullptr) {
        *conv_bias_md = mkldnn::memory::desc({ conv_bias_tz }, type, mkldnn::memory::format::any);
        *user_bias_md = mkldnn::memory::desc({ conv_bias_tz }, type, mkldnn::memory::format::x);
    }

    if (dst != nullptr && conv_dst_md != nullptr) {
        *conv_dst_md = mkldnn::memory::desc({ conv_dst_tz }, type, mkldnn::memory::format::any);
        *user_dst_md = mkldnn::memory::desc({ conv_dst_tz }, type, format);
        user_dst_md->data.format = mkldnn_blocked; // overrides "format = isNCDHW ? ncdhw : ndhwc"
        user_dst_md->data.layout_desc.blocking.strides[0][0] = dst->stridesOf()[isNCDHW ? 0 : 0];
        user_dst_md->data.layout_desc.blocking.strides[0][1] = dst->stridesOf()[isNCDHW ? 1 : 4];
        user_dst_md->data.layout_desc.blocking.strides[0][2] = dst->stridesOf()[isNCDHW ? 2 : 1];
        user_dst_md->data.layout_desc.blocking.strides[0][3] = dst->stridesOf()[isNCDHW ? 3 : 2];
        user_dst_md->data.layout_desc.blocking.strides[0][4] = dst->stridesOf()[isNCDHW ? 4 : 3];
    }
}
#endif

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void conv2d_(nd4j::LaunchContext & block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

    // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    // weights [kH, kW, iC, oC] always
    // bias    [oC]
    // output  [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

    // kH  filter(kernel) height
    // kW  filter(kernel) width
    // sH  strides height
    // sW  strides width
    // pH  paddings height
    // pW  paddings width
    // dH  dilations height
    // dW  dilations width
    // isSameMode 0-VALID, 1-SAME
    // isNCHW     1-NCHW,  0-NHWC

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    if(isSameMode)                       // SAME
        ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

#ifdef HAVE_MKLDNN
    if (block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported<X, Y>()) {
        std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
        if (streams.empty()) {
            streams.push_back(MKLDNNStream("conv2d"));
        }

        if (streams[0].checkAndReset({input, weights, bias}, {output}, {}, {kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW})) {
            mkldnn_memory_desc_t empty;
            mkldnn::memory::desc conv_src_md(empty), conv_weights_md(empty), conv_bias_md(empty), conv_dst_md(empty);
            mkldnn::memory::desc user_src_md(empty), user_weights_md(empty), user_bias_md(empty), user_dst_md(empty);
            mkldnn::memory::dims conv_strides, conv_padding, conv_padding_r;

            ConvolutionUtils::getMKLDNNMemoryDescConv2d(kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW,
                    bS, iC, iH, iW, oC, oH, oW, input, nullptr, weights, nullptr, bias, output,
                    &conv_src_md, nullptr, &conv_weights_md, nullptr, &conv_bias_md, &conv_dst_md,
                    &user_src_md, nullptr, &user_weights_md, nullptr, &user_bias_md, &user_dst_md,
                    conv_strides, conv_padding, conv_padding_r);

            auto conv_desc = bias != nullptr
                    ? convolution_forward::desc(prop_kind::forward,
                            convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
                            conv_dst_md, conv_strides, conv_padding, conv_padding_r, padding_kind::zero)
                    : convolution_forward::desc(prop_kind::forward,
                            convolution_direct, conv_src_md, conv_weights_md,
                            conv_dst_md, conv_strides, conv_padding, conv_padding_r, padding_kind::zero);

            auto engine = streams[0].getEngine();
            auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, engine);
            auto user_src_memory = mkldnn::memory({user_src_md, engine}, const_cast<NDArray*>(input)->buffer());
            auto user_weights_memory = mkldnn::memory({user_weights_md, engine}, const_cast<NDArray*>(weights)->buffer());
            auto user_dst_memory = mkldnn::memory({user_dst_md, engine}, output->buffer());

            auto conv_src_memory = user_src_memory;
            streams[0].addMemory(user_src_memory);
            if (mkldnn::memory::primitive_desc(conv_prim_desc.src_primitive_desc())
                    != user_src_memory.get_primitive_desc()) {
                conv_src_memory = mkldnn::memory(conv_prim_desc.src_primitive_desc());
                streams[0].addMemory(conv_src_memory);
                streams[0].addOperation(reorder(user_src_memory, conv_src_memory));
            }

            auto conv_weights_memory = user_weights_memory;
            streams[0].addMemory(user_weights_memory);
            if (mkldnn::memory::primitive_desc(conv_prim_desc.weights_primitive_desc())
                    != user_weights_memory.get_primitive_desc()) {
                conv_weights_memory = mkldnn::memory(conv_prim_desc.weights_primitive_desc());
                streams[0].addMemory(conv_weights_memory);
                streams[0].addOperation(reorder(user_weights_memory, conv_weights_memory));
            }

            auto conv_dst_memory = user_dst_memory;
            streams[0].addMemory(user_dst_memory);
            if (mkldnn::memory::primitive_desc(conv_prim_desc.dst_primitive_desc())
                    != user_dst_memory.get_primitive_desc()) {
                conv_dst_memory = mkldnn::memory(conv_prim_desc.dst_primitive_desc());
                streams[0].addMemory(conv_dst_memory);
            }

            if (bias != nullptr) {
                auto conv_bias_memory = mkldnn::memory(conv_prim_desc.bias_primitive_desc(), const_cast<NDArray*>(bias)->buffer());
                streams[0].addMemory(conv_bias_memory);
                streams[0].addOperation(convolution_forward(conv_prim_desc, conv_src_memory, conv_weights_memory, conv_bias_memory, conv_dst_memory));
            } else {
                streams[0].addOperation(convolution_forward(conv_prim_desc, conv_src_memory, conv_weights_memory, conv_dst_memory));
            }

            if (mkldnn::memory::primitive_desc(conv_prim_desc.dst_primitive_desc())
                    != user_dst_memory.get_primitive_desc()) {
                streams[0].addOperation(reorder(conv_dst_memory, user_dst_memory));
            }
        }

        streams[0].submitAndWait();
        return;
    }
#endif
    nd4j_debug("MKL-DNN is not used for conv2d!\n", 0);

    std::vector<int> permutForOutput;
    if(!isNCHW)
        input = input->permute({0, 3, 1, 2});                                       // [bS, iH, iW, iC] -> [bS, iC, iH, iW] if NHWC
    else
        // permutForOutput = {0, indOoH, indOoH+1, indIOioC};                          // [bS, oC, oH, oW] -> [bS, oH, oW, oC]
        permutForOutput = {0, 3, 1, 2};                                             // [bS, oH, oW, oC] -> [bS, oC, oH, oW]

    NDArray col('c', {bS, oH, oW, kH, kW, iC}, input->dataType(), &block);
    NDArray* colP = col.permute({0, 5, 3, 4, 1, 2});            // {bS, iC, kH, kW, oH, oW}
    NDArray mmulResult('f', {bS*oH*oW, oC}, output->dataType(), &block);

    //----- calculation of output -----//
    nd4j::LaunchContext  ctx;
    helpers::im2col(ctx, *input, *colP, kH, kW, sH, sW, pH, pW, dH, dW, NDArrayFactory::create(0.f, &block));  // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
    MmulHelper::tensorDot(&col, weights, &mmulResult, {3,4,5}, {0,1,2}, {}); // [bS, oH, oW, kH, kW, iC] x [kH, kW, iC, oC] = [bS, oH, oW, oC]

    //----- assign outTemp to output  -----//
    if(isNCHW) {
        mmulResult.reshapei({bS, oH, oW, oC});
        mmulResult.permutei(permutForOutput);
    }
    output->assign(mmulResult);

    //----- add biases if required -----//
    if(bias)
        // output->applyBroadcast(broadcast::Add, {indIOioC}, bias);
        helpers::addBias(*output, *bias, isNCHW);

    if(!isNCHW)
        delete input;

    delete colP;
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void conv2dBP_(nd4j::LaunchContext & block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

    // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    // weights [kH, kW, iC, oC] always
    // bias    [oC]
    // gradO   [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next

    // gradI    [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
    // gradW    [kH, kW, iC, oC] always
    // gradB    [oC]

    // kH         filter(kernel) height
    // kW         filter(kernel) width
    // sH         strides height
    // sW         strides width
    // pH         paddings height
    // pW         paddings width
    // dH         dilations height
    // dW         dilations width
    // isSameMode 0-VALID, 1-SAME
    // isNCHW     0-NHWC, 1-NCHW

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    if(isSameMode)                       // SAME
        ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

#ifdef HAVE_MKLDNN
    if (block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported<X, Y>()) {
        std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
        if (streams.empty()) {
            streams.push_back(MKLDNNStream("conv2d_bp_weights"));
            streams.push_back(MKLDNNStream("conv2d_bp_data"));
        }

        bool resetW = streams[0].checkAndReset({input, weights, bias, gradO}, {gradI, gradW, gradB}, {}, {kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW});
        bool resetI = streams[1].checkAndReset({input, weights, bias, gradO}, {gradI, gradW, gradB}, {}, {kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW});
        if (resetW || resetI) {
            mkldnn_memory_desc_t empty;
            mkldnn::memory::desc conv_src_md(empty), conv_diff_src_md(empty), conv_weights_md(empty),
                                 conv_diff_weights_md(empty), conv_bias_md(empty), conv_dst_md(empty);
            mkldnn::memory::desc user_src_md(empty), user_diff_src_md(empty), user_weights_md(empty),
                                 user_diff_weights_md(empty), user_bias_md(empty), user_dst_md(empty);
            mkldnn::memory::dims conv_strides, conv_padding, conv_padding_r;

            ConvolutionUtils::getMKLDNNMemoryDescConv2d(kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW,
                    bS, iC, iH, iW, oC, oH, oW, input, gradI, weights, gradW, gradB, gradO,
                    &conv_src_md, &conv_diff_src_md, &conv_weights_md, &conv_diff_weights_md, &conv_bias_md, &conv_dst_md,
                    &user_src_md, &user_diff_src_md, &user_weights_md, &user_diff_weights_md, &user_bias_md, &user_dst_md,
                    conv_strides, conv_padding, conv_padding_r);

            auto conv_desc = gradB != nullptr
                    ? convolution_forward::desc(prop_kind::forward,
                            convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
                            conv_dst_md, conv_strides, conv_padding, conv_padding_r, padding_kind::zero)
                    : convolution_forward::desc(prop_kind::forward,
                            convolution_direct, conv_src_md, conv_weights_md,
                            conv_dst_md, conv_strides, conv_padding, conv_padding_r, padding_kind::zero);

            auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, streams[0].getEngine());

            if (gradW != nullptr) {
                auto convW_desc = gradB != nullptr
                        ? convolution_backward_weights::desc(
                                convolution_direct, conv_src_md, conv_diff_weights_md, conv_bias_md,
                                conv_dst_md, conv_strides, conv_padding, conv_padding_r, padding_kind::zero)
                        : convolution_backward_weights::desc(
                                convolution_direct, conv_src_md, conv_diff_weights_md,
                                conv_dst_md, conv_strides, conv_padding, conv_padding_r, padding_kind::zero);

                auto engine = streams[0].getEngine();
                auto convW_prim_desc = convolution_backward_weights::primitive_desc(convW_desc, engine, conv_prim_desc);
                auto userW_src_memory = mkldnn::memory({user_src_md, engine}, const_cast<NDArray*>(input)->buffer());
                auto userW_weights_memory = mkldnn::memory({user_diff_weights_md, engine}, gradW->buffer());
                auto userW_dst_memory = mkldnn::memory({user_dst_md, engine}, const_cast<NDArray*>(gradO)->buffer());

                auto convW_src_memory = userW_src_memory;
                streams[0].addMemory(userW_src_memory);
                if (mkldnn::memory::primitive_desc(convW_prim_desc.src_primitive_desc())
                        != userW_src_memory.get_primitive_desc()) {
                    convW_src_memory = mkldnn::memory(convW_prim_desc.src_primitive_desc());
                    streams[0].addMemory(convW_src_memory);
                    streams[0].addOperation(reorder(userW_src_memory, convW_src_memory));
                }

                auto convW_weights_memory = userW_weights_memory;
                streams[0].addMemory(userW_weights_memory);
                if (mkldnn::memory::primitive_desc(convW_prim_desc.diff_weights_primitive_desc())
                        != userW_weights_memory.get_primitive_desc()) {
                    convW_weights_memory = mkldnn::memory(convW_prim_desc.diff_weights_primitive_desc());
                    streams[0].addMemory(convW_weights_memory);
                }

                auto convW_dst_memory = userW_dst_memory;
                streams[0].addMemory(userW_dst_memory);
                if (mkldnn::memory::primitive_desc(convW_prim_desc.diff_dst_primitive_desc())
                        != userW_dst_memory.get_primitive_desc()) {
                    convW_dst_memory = mkldnn::memory(convW_prim_desc.diff_dst_primitive_desc());
                    streams[0].addMemory(convW_dst_memory);
                    streams[0].addOperation(reorder(userW_dst_memory, convW_dst_memory));
                }

                if (gradB != nullptr) {
                    auto convW_bias_memory = mkldnn::memory(convW_prim_desc.diff_bias_primitive_desc(), gradB->buffer());
                    streams[0].addMemory(convW_bias_memory);
                    streams[0].addOperation(convolution_backward_weights(convW_prim_desc, convW_src_memory, convW_dst_memory, convW_weights_memory, convW_bias_memory));
                } else {
                    streams[0].addOperation(convolution_backward_weights(convW_prim_desc, convW_src_memory, convW_dst_memory, convW_weights_memory));
                }

                if (mkldnn::memory::primitive_desc(convW_prim_desc.diff_weights_primitive_desc())
                        != userW_weights_memory.get_primitive_desc()) {
                    streams[0].addOperation(reorder(convW_weights_memory, userW_weights_memory));
                }
            }

            if (gradI != nullptr) {
                auto convI_desc =
                        convolution_backward_data::desc(
                                convolution_direct, conv_diff_src_md, conv_weights_md,
                                conv_dst_md, conv_strides, conv_padding, conv_padding_r, padding_kind::zero);

                auto engine = streams[1].getEngine();
                auto convI_prim_desc = convolution_backward_data::primitive_desc(convI_desc, engine, conv_prim_desc);
                auto userI_src_memory = mkldnn::memory({user_diff_src_md, engine}, gradI->buffer());
                auto userI_weights_memory = mkldnn::memory({user_weights_md, engine}, const_cast<NDArray*>(weights)->buffer());
                auto userI_dst_memory = mkldnn::memory({user_dst_md, engine}, const_cast<NDArray*>(gradO)->buffer());

                auto convI_src_memory = userI_src_memory;
                streams[1].addMemory(userI_src_memory);
                if (mkldnn::memory::primitive_desc(convI_prim_desc.diff_src_primitive_desc())
                        != userI_src_memory.get_primitive_desc()) {
                    convI_src_memory = mkldnn::memory(convI_prim_desc.diff_src_primitive_desc());
                    streams[1].addMemory(convI_src_memory);
                }

                auto convI_weights_memory = userI_weights_memory;
                streams[1].addMemory(userI_weights_memory);
                if (mkldnn::memory::primitive_desc(convI_prim_desc.weights_primitive_desc())
                        != userI_weights_memory.get_primitive_desc()) {
                    convI_weights_memory = mkldnn::memory(convI_prim_desc.weights_primitive_desc());
                    streams[1].addMemory(convI_weights_memory);
                    streams[1].addOperation(reorder(userI_weights_memory, convI_weights_memory));
                }

                auto convI_dst_memory = userI_dst_memory;
                streams[1].addMemory(userI_dst_memory);
                if (mkldnn::memory::primitive_desc(convI_prim_desc.diff_dst_primitive_desc())
                        != userI_dst_memory.get_primitive_desc()) {
                    convI_dst_memory = mkldnn::memory(convI_prim_desc.diff_dst_primitive_desc());
                    streams[1].addMemory(convI_dst_memory);
                    streams[1].addOperation(reorder(userI_dst_memory, convI_dst_memory));
                }

                streams[1].addOperation(convolution_backward_data(convI_prim_desc, convI_dst_memory, convI_weights_memory, convI_src_memory));

                if (mkldnn::memory::primitive_desc(convI_prim_desc.diff_src_primitive_desc())
                        != userI_src_memory.get_primitive_desc()) {
                    streams[1].addOperation(reorder(convI_src_memory, userI_src_memory));
                }
            }
        }

        if (gradW != nullptr) {
            streams[0].submitAndWait();
        }
        if (gradI != nullptr) {
            streams[1].submitAndWait();
        }
        return;
    }
#endif
    nd4j_debug("MKL-DNN is not used for conv2d_bp!\n", 0);

    std::vector<int> gradOaxesForDot;

    if(!isNCHW) {
        input = input->permute({0, 3, 1, 2});                                   // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
        gradI = gradI->permute({0, 3, 1, 2});                                   // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
        gradOaxesForDot  = {0, 1, 2};                                           // bS, oH, oW
    }
    else
        gradOaxesForDot  = {0, 2, 3};                                           // bS, oH, oW

    NDArray columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->dataType(), input->getContext());

    // ----- calculation of gradW ----- //
    if(gradW) {
        nd4j::LaunchContext * ctx = &block;
        helpers::im2col(*ctx, *input, columns, kH, kW, sH, sW, pH, pW, dH, dW, NDArrayFactory::create(0.f, input->getContext()));   // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
        nd4j::MmulHelper::tensorDot(&columns, gradO, gradW, {0,4,5}, gradOaxesForDot, {2, 0, 1, 3});       // [bS, iC, kH, kW, oH, oW] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [iC, kH, kW, oC]
    }

    // ----- calculation of gradB ----- //
    if(gradB) {
        NDArray* gradBR = gradB;
        if(gradB->rankOf() == 2)
            gradBR = gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()});
        gradO->reduceAlongDimension(reduce::Sum, gradBR, gradOaxesForDot);                          // sum over bS, oH, oW
        if(gradBR != gradB)
            delete gradBR;
    }

    //----- calculation of gradI -----//
    nd4j::MmulHelper::tensorDot(weights, gradO, &columns, {indWoC}, {indIOioC}, {2, 3, 1, 0, 4, 5});  // [kH, kW, iC, oC]/[oC, iC, kH, kW]] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [kH, kW, iC, bS, oH, oW]
    nd4j::LaunchContext * ctx = &block;
    helpers::col2im(*ctx, columns, *gradI, sH, sW, pH, pW, iH, iW, dH, dW);                          // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]

    if(!isNCHW) {
        delete input;
        delete gradI;
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void depthwiseConv2d_(nd4j::LaunchContext & block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

    // input     [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    // weights   [kH, kW, iC, mC] always
    // bias      [oC] = iC*mC
    // output    [bS, oH, oW, iC*mC] (NHWC) or [bS, iC*mC, oH, oW] (NCHW)

    // kH           filter(kernel) height
    // kW           filter(kernel) width
    // sH           strides height
    // sW           strides width
    // pH           paddings height
    // pW           paddings width
    // dH           dilations height
    // dW           dilations width
    // isSameMode   0-VALID, 1-SAME
    // isNCHW       0-NCHW,  1-NHWC

    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);
    mC = weights->sizeAt(indWmC);                           // channels multiplier

    std::vector<std::vector<Nd4jLong>> modifColumns = {{1,0,4,5,2,3}, {iC,bS*oH*oW,kH*kW}};  // [bS,iC,kH,kW,oH,oW] -> [iC,bS,oH,oW,kH,kW] -> [iC,bS*oH*oW,kH*kW]
    std::vector<std::vector<Nd4jLong>> modifOutput;
    std::vector<Nd4jLong> outReShape;

    if(!isNCHW) {
        input = input->permute({0, 3, 1, 2});                                           // [bS,iH,iW,iC]    -> [bS,iC,iH,iW]
        outReShape = {bS, oH, oW, iC, mC};                                              // [bS,oH,oW,iC*mC] -> [bS,oH,oW,iC,mC]
        modifOutput = {{3,0,1,2,4},{iC, bS*oH*oW, mC}};                                 // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
    }
    else {
        outReShape = {bS, iC, mC, oH, oW};                                              // [bS,iC*mC,oH,oW] -> [bS,iC,mC,oH,oW]
        modifOutput = {{1,0,3,4,2},{iC, bS*oH*oW, mC}};                                 // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
    }

    if(isSameMode)                       // SAME
        ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->dataType(), input->getContext());
    NDArray* outputReshaped = output->reshape(output->ordering(), outReShape);

    nd4j::LaunchContext * ctx = &block;
    helpers::im2col(*ctx, *input, columns, kH, kW, sH, sW, pH, pW, dH, dW, NDArrayFactory::create(0.f, input->getContext()));  // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
    MmulHelper::tensorDot(&columns, weights, outputReshaped, modifColumns, {{2,0,1,3},{iC,kH*kW,mC}}, modifOutput);              // [iC, bS*oH*oW, kW*kH] x [iC, kH*kW, mC] = [iC, bS*oH*oW, mC]

    if(bias)
        output->applyBroadcast(broadcast::Add, {indIOioC}, bias);

    if(!isNCHW)
        delete input;

    delete outputReshaped;
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void depthwiseConv2dBP_(nd4j::LaunchContext & block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

    // input    [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW)
    // weights  [kH, kW, iC, mC] always
    // bias     [oC] = [iC*mC]
    // gradO    [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next
    // gradI    [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW), epsilon
    // gradW    [kH, kW, iC, mC] always
    // gradB    [oC]

    //  kH          filter(kernel) height
    //  kW          filter(kernel) width
    //  sH          strides height
    //  sW          strides width
    //  pH          paddings height
    //  pW          paddings width
    //  dH          dilations height
    //  dW          dilations width
    //  isSameMode  0-VALID, 1-SAME
    //  isNCHW      0-NHWC, 1-NCHW

    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);
    mC = weights->sizeAt(indWmC);                           // channels multiplier

    std::vector<std::vector<Nd4jLong>> modifColumns = {{1,2,3,0,4,5}, {iC, kH*kW, bS*oH*oW}};      // [bS,iC,kH,kW,oH,oW] -> [iC, kH*kW, bS*oH*oW]
    std::vector<std::vector<Nd4jLong>> modifGradO1, modifGradO2;
    std::vector<Nd4jLong> gradOreShape;

    if(!isNCHW) {
        input = input->permute({0, 3, 1, 2});                                           // [bS,iH,iW,iC]    -> [bS,iC,iH,iW]
        gradI = gradI->permute({0, 3, 1, 2});                                           // [bS,iH,iW,iC]    -> [bS,iC,iH,iW]
        gradOreShape = {bS, oH, oW, iC, mC};                                            // [bS,oH,oW,iC*mC] -> [bS,oH,oW,iC,mC]
        modifGradO1 = {{3,0,1,2,4},{iC, bS*oH*oW, mC}};                                 // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifGradO2 = {{3,0,1,2},{iC, mC, bS*oH*oW}};                                   // [bS,oH,oW,iC*mC] -> [iC*mC,bS,oH,oW] -> [iC,mC,bS*oH*oW]
    }
    else {
        gradOreShape = {bS, iC, mC, oH, oW};                                            // [bS,iC*mC,oH,oW] -> [bS,iC,mC,oH,oW]
        modifGradO1 = {{1,0,3,4,2},{iC, bS*oH*oW, mC}};                                 // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifGradO2 = {{1,0,2,3},{iC, mC, bS*oH*oW}};                                   // [bS,iC*mC,oH,oW] -> [iC*mC,bS,oH,oW] -> [iC,mC,bS*oH*oW]
    }

    if(isSameMode)                       // SAME
        ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray  columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->dataType(), input->getContext());
    NDArray* gradOreshaped = gradO->reshape(gradO->ordering(), gradOreShape);

    // ----- calculation of gradW and gradB ----- //

    nd4j::LaunchContext * ctx = &block;
    helpers::im2col(*ctx, *input, columns, kH, kW, sH, sW, pH, pW, dH, dW, NDArrayFactory::create(0.f, input->getContext()));  // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
    nd4j::MmulHelper::tensorDot(&columns, gradOreshaped, gradW, modifColumns, modifGradO1, {{2,0,1,3},{iC,kH*kW,mC}});  // [iC, kW*kH, bS*oH*oW] x [iC, bS*oH*oW, mC] = [iC, kH*kW, mC]

    // ----- calculation of gradB ----- //
    if(gradB) {
        NDArray* gradBR = gradB;
        if(gradB->rankOf() == 2)
            gradBR = gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()});
        gradO->reduceAlongDimension(reduce::Sum, gradBR, {0,indOoH,indOoH+1});                      // sum over bS, oH, oW
        if(gradBR != gradB)
            delete gradBR;
    }

    //----- calculation of gradI -----//
    nd4j::MmulHelper::tensorDot(weights, gradO, &columns, {{2,0,1,3},{iC,kH*kW,mC}}, modifGradO2, modifColumns); // [iC, kH*kW, mC] x [iC, mC, bS*oH*oW] = [iC, kW*kH, bS*oH*oW]
    helpers::col2im(*ctx, columns, *gradI, sH, sW, pH, pW, iH, iW, dH, dW);                                       // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]

    if(!isNCHW) {
        delete input;
        delete gradI;
    }

    delete gradOreshaped;
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void sconv2d_(nd4j::LaunchContext & block, const NDArray* input, const NDArray* weightsDepth, const NDArray* weightsPoint, const NDArray* bias,  NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

    // input         [bS, iH, iW, iC]  (NHWC) or [bS, iC, iH, iW]  (NCHW)
    // weightsDepth  [kH, kW, iC, mC]  always
    // weightsPoint  [1, 1, iC*mC, oC] always
    // bias          [oC], oC = iC*mC if weightsPoint=nullptr
    // output is     [bS, oH, oW, oC]  (NHWC) or [bS, oC, oH, oW]  (NCHW)

    //  kH         filter(kernel) height
    //  kW         filter(kernel) width
    //  sH         strides height
    //  sW         strides width
    //  pH         paddings height
    //  pW         paddings width
    //  dH         dilations height
    //  dW         dilations width
    //  isSameMode 0-VALID, 1-SAME
    //  isNCHW     1-NCHW,  0-NHWC

    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier, output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);
    mC = weightsDepth->sizeAt(indWmC);                      // channels multiplier

    NDArray* outputDepth = output;
    if(weightsPoint)                        // if pointwise convolution is expected
        outputDepth = new NDArray(output->ordering(), !isNCHW ? std::vector<Nd4jLong>({bS, oH, oW, iC*mC}) : std::vector<Nd4jLong>({bS, iC*mC, oH, oW}), input->dataType(), input->getContext());

    // ----- perform depthwise convolution (if weightsPoint is absent then oC = iC*mC) ----- //
    ConvolutionUtils::depthwiseConv2d(block, input, weightsDepth, weightsPoint ? nullptr : bias, outputDepth, kH,kW, sH,sW, pH,pW, dH,dW, isSameMode, isNCHW);

    // ----- perform pointwise convolution (oH = iH, oW = iW) ----- //
    if (weightsPoint) {
        ConvolutionUtils::conv2d(block, outputDepth, weightsPoint, bias, output, 1,1, 1,1, 0,0, 1,1, isSameMode, isNCHW);             // in this case oH=iH, oW=iW
        delete outputDepth;
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void upsampling2d_(const NDArray& input, NDArray& output, const int factorH, const int factorW, const bool isNCHW) {
    // input  has shape [bS, iC, iH, iW] (NCHW) or [bS, iH, iW, iC] (NHWC) 
    // output has shape [bS, iC, factorH*iH, factorW*iW ] (NCHW) or [bS, factorH*iH, factorW*iW, iC] (NHWC)
    
    std::vector<Nd4jLong> indIn  = {0,0,  0,0,  0,0,  0,0};
    std::vector<Nd4jLong> indOut = {0,0,  0,0,  0,0,  0,0};
    const int dimIH = isNCHW ? 2 : 1;    
    const int j0 = 2*dimIH;
    const int j1 = j0+1, j2 = j0+2, j3 = j0+3;
    const int size0 = input.sizeAt(dimIH) * input.sizeAt(dimIH+1);
    // const int size1 = factorH * factorW;

    int iT = input.sizeAt(dimIH);
    int iH = input.sizeAt(dimIH + 1);

    PRAGMA_OMP_PARALLEL_FOR_ARGS(collapse(2) firstprivate(indIn, indOut))
    for(int ih = 0; ih < iT; ++ih) {
        for(int iw = 0; iw < iH; ++iw) {
            indIn[j0] = ih; indIn[j1] = ih+1; 
            indIn[j2] = iw; indIn[j3] = iw+1; 

            for(int fh = 0; fh < factorH; ++fh) {
                for(int fw = 0; fw < factorW; ++fw) {
                    
                    indOut[j0] = ih * factorH + fh; indOut[j1] = indOut[j0] + 1; 
                    indOut[j2] = iw * factorW + fw; indOut[j3] = indOut[j2] + 1;                     
                    auto i = input(indIn);
                    auto o = output(indOut);
                    o.assign(i);
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void upsampling3d_(const NDArray& input, NDArray& output, const int factorD, const int factorH, const int factorW, const bool isNCDHW) {
    // input  has shape [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC) 
    // output has shape [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH, factorW*iW, iC] (NDHWC)
    std::vector<Nd4jLong> indIn  = {0,0,  0,0,  0,0,  0,0,  0,0};
    std::vector<Nd4jLong> indOut = {0,0,  0,0,  0,0,  0,0,  0,0};
    const int dimID = isNCDHW ? 2 : 1;    
    const int j0 = 2*dimID;
    const int j1 = j0+1, j2 = j0+2, j3 = j0+3, j4 = j0+4, j5 = j0+5;;
    const int size0 = input.sizeAt(dimID) * input.sizeAt(dimID+1) * input.sizeAt(dimID+2);
    // const int size1 = factorD * factorH * factorW;

    int l0 = input.sizeAt(dimID);
    int l1 = input.sizeAt(dimID + 1);
    int l2 = input.sizeAt(dimID + 2);

    PRAGMA_OMP_PARALLEL_FOR_ARGS(collapse(2) firstprivate(indIn, indOut))
    for(int id = 0; id < l0; ++id) {
        for(int ih = 0; ih < l1; ++ih) {
            for(int iw = 0; iw < l2; ++iw) {
                indIn[j0] = id; indIn[j1] = id+1;
                indIn[j2] = ih; indIn[j3] = ih+1;
                indIn[j4] = iw; indIn[j5] = iw+1;

            for(int fd = 0; fd < factorD; ++fd) {
                for(int fh = 0; fh < factorH; ++fh) {
                    for(int fw = 0; fw < factorW; ++fw) {
                            indOut[j0] = id * factorD + fd; indOut[j1] = indOut[j0] + 1; 
                            indOut[j2] = ih * factorH + fh; indOut[j3] = indOut[j2] + 1; 
                            indOut[j4] = iw * factorW + fw; indOut[j5] = indOut[j4] + 1;                     
                            auto i = input(indIn);                    
                            auto o = output(indOut);
                            o.assign(i);
                        }
                    }
                }
            }
        }
    }    
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void upsampling2dBP_(const NDArray& gradO, NDArray& gradI, const bool isNCHW) {
    // gradO has shape [bS, iC, factorH*iH, factorW*iW ] (NCHW) or [bS, factorH*iH, factorW*iW, iC] (NHWC)
    // gradI has shape [bS, iC, iH, iW] (NCHW) or [bS, iH, iW, iC] (NHWC)     
    std::vector<Nd4jLong> indIn  = {0,0,  0,0,  0,0,  0,0};
    std::vector<Nd4jLong> indOut = {0,0,  0,0,  0,0,  0,0};
    const int dimIH = isNCHW ? 2 : 1;    
    const int factorH = gradO.sizeAt(dimIH)   / gradI.sizeAt(dimIH);
    const int factorW = gradO.sizeAt(dimIH+1) / gradI.sizeAt(dimIH+1);
    const int j0 = 2*dimIH;
    const int j1 = j0+1, j2 = j0+2, j3 = j0+3;
    const int size0 = gradI.sizeAt(dimIH) * gradI.sizeAt(dimIH+1);

    int l0 = gradI.sizeAt(dimIH);
    int l1 = gradI.sizeAt(dimIH + 1);

    PRAGMA_OMP_PARALLEL_FOR_ARGS(collapse(2) firstprivate(indIn, indOut))
    for(int ih = 0; ih < l0; ++ih) {
        for(int iw = 0; iw < l1; ++iw) {
            indIn[j0] = ih; indIn[j1] = ih+1; 
            indIn[j2] = iw; indIn[j3] = iw+1; 
            NDArray subGradI = gradI(indIn);

            for(int fh = 0; fh < factorH; ++fh) {
                for(int fw = 0; fw < factorW; ++fw) {                    
                    indOut[j0] = ih * factorH + fh; indOut[j1] = indOut[j0] + 1; 
                    indOut[j2] = iw * factorW + fw; indOut[j3] = indOut[j2] + 1;                     
                    auto o = gradO(indOut);
                    if(!fh && !fw) {                        
                        subGradI.assign(o);
                    }
                    else
                        subGradI += o;
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void upsampling3dBP_(const NDArray& gradO, NDArray& gradI, const bool isNCDHW) {
    // input  has shape [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC) 
    // output has shape [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH, factorW*iW, iC] (NDHWC)
    std::vector<Nd4jLong> indIn  = {0,0,  0,0,  0,0,  0,0,  0,0};
    std::vector<Nd4jLong> indOut = {0,0,  0,0,  0,0,  0,0,  0,0};
    const int dimID = isNCDHW ? 2 : 1;
    const int factorD = gradO.sizeAt(dimID)   / gradI.sizeAt(dimID);
    const int factorH = gradO.sizeAt(dimID+1) / gradI.sizeAt(dimID+1);
    const int factorW = gradO.sizeAt(dimID+2) / gradI.sizeAt(dimID+2);
    const int j0 = 2*dimID;
    const int j1 = j0+1, j2 = j0+2, j3 = j0+3, j4 = j0+4, j5 = j0+5;;
    const int size0 = gradI.sizeAt(dimID) * gradI.sizeAt(dimID+1) * gradI.sizeAt(dimID+2);

    int l0 = gradI.sizeAt(dimID);
    int l1 = gradI.sizeAt(dimID + 1);
    int l2 = gradI.sizeAt(dimID + 2);

    PRAGMA_OMP_PARALLEL_FOR_ARGS(collapse(3) firstprivate(indOut, indIn))
    for(int id = 0; id < l0; ++id) {
        for(int ih = 0; ih < l1; ++ih) {
            for(int iw = 0; iw < l2; ++iw) {
                indIn[j0] = id; indIn[j1] = id+1;
                indIn[j2] = ih; indIn[j3] = ih+1;
                indIn[j4] = iw; indIn[j5] = iw+1;
                NDArray subGradI = gradI(indIn);

            for(int fd = 0; fd < factorD; ++fd) {
                for(int fh = 0; fh < factorH; ++fh) {
                    for(int fw = 0; fw < factorW; ++fw) {
                            indOut[j0] = id * factorD + fd; indOut[j1] = indOut[j0] + 1; 
                            indOut[j2] = ih * factorH + fh; indOut[j3] = indOut[j2] + 1; 
                            indOut[j4] = iw * factorW + fw; indOut[j5] = indOut[j4] + 1;                     
                            auto o = gradO(indOut);
                            if(!fd && !fh && !fw)
                                subGradI.assign(o);
                            else
                                subGradI += o;
                        }
                    }
                }
            }
        }
    }    
}


#ifdef HAVE_MKLDNN
using namespace mkldnn;

void ConvolutionUtils::getMKLDNNMemoryDescPool2d(
        int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int poolingMode, int extraParam0, bool isNCHW,
        int bS, int iC, int iH, int iW, int oC, int oH, int oW,
        const NDArray* src, const NDArray* diff_src, const NDArray* dst, mkldnn::algorithm& algorithm,
        mkldnn::memory::desc* pool_src_md, mkldnn::memory::desc* pool_diff_src_md, mkldnn::memory::desc* pool_dst_md,
        mkldnn::memory::desc* user_src_md, mkldnn::memory::desc* user_diff_src_md, mkldnn::memory::desc* user_dst_md,
        mkldnn::memory::dims& pool_strides, mkldnn::memory::dims& pool_kernel, mkldnn::memory::dims& pool_padding, mkldnn::memory::dims& pool_padding_r) {
    mkldnn::memory::dims pool_src_tz = { bS, iC, iH, iW };
    mkldnn::memory::dims pool_dst_tz = { bS, oC, oH, oW };

    pool_strides = { sH, sW };
    pool_kernel = { kH, kW };
    pool_padding = { pH, pW };
    pool_padding_r = { (oH - 1) * sH - iH + kH - pH,
                       (oW - 1) * sW - iW + kW - pW };

    algorithm = poolingMode == 0 ? pooling_max
                                 : extraParam0 == 0 ? pooling_avg_exclude_padding
                                                    : pooling_avg_include_padding;
    auto type = mkldnn::memory::data_type::f32;
    auto format = isNCHW ? mkldnn::memory::format::nchw : mkldnn::memory::format::nhwc;
    auto supposed_to_be_any_format = mkldnn::memory::format::nChw8c; // doesn't work with "any"

    if (src != nullptr && src->getBuffer() != nullptr && pool_src_md != nullptr) {
        *pool_src_md = mkldnn::memory::desc({ pool_src_tz }, type, supposed_to_be_any_format);
        *user_src_md = mkldnn::memory::desc({ pool_src_tz }, type, format);
        user_src_md->data.format = mkldnn_blocked; // overrides "format = isNCHW ? nchw : nhwc"
        user_src_md->data.layout_desc.blocking.strides[0][0] = src->stridesOf()[isNCHW ? 0 : 0];
        user_src_md->data.layout_desc.blocking.strides[0][1] = src->stridesOf()[isNCHW ? 1 : 3];
        user_src_md->data.layout_desc.blocking.strides[0][2] = src->stridesOf()[isNCHW ? 2 : 1];
        user_src_md->data.layout_desc.blocking.strides[0][3] = src->stridesOf()[isNCHW ? 3 : 2];
    }

    if (diff_src != nullptr && diff_src->getBuffer() != nullptr && pool_diff_src_md != nullptr) {
        *pool_diff_src_md = mkldnn::memory::desc({ pool_src_tz }, type, supposed_to_be_any_format);
        *user_diff_src_md = mkldnn::memory::desc({ pool_src_tz }, type, format);
        user_diff_src_md->data.format = mkldnn_blocked; // overrides "format = isNCHW ? nchw : nhwc"
        user_diff_src_md->data.layout_desc.blocking.strides[0][0] = diff_src->stridesOf()[isNCHW ? 0 : 0];
        user_diff_src_md->data.layout_desc.blocking.strides[0][1] = diff_src->stridesOf()[isNCHW ? 1 : 3];
        user_diff_src_md->data.layout_desc.blocking.strides[0][2] = diff_src->stridesOf()[isNCHW ? 2 : 1];
        user_diff_src_md->data.layout_desc.blocking.strides[0][3] = diff_src->stridesOf()[isNCHW ? 3 : 2];
    }

    if (dst != nullptr && dst->getBuffer() != nullptr && pool_dst_md != nullptr) {
        *pool_dst_md = mkldnn::memory::desc({ pool_dst_tz }, type, supposed_to_be_any_format);
        *user_dst_md = mkldnn::memory::desc({ pool_dst_tz }, type, format);
        user_dst_md->data.format = mkldnn_blocked; // overrides "format = isNCHW ? nchw : nhwc"
        user_dst_md->data.layout_desc.blocking.strides[0][0] = dst->stridesOf()[isNCHW ? 0 : 0];
        user_dst_md->data.layout_desc.blocking.strides[0][1] = dst->stridesOf()[isNCHW ? 1 : 3];
        user_dst_md->data.layout_desc.blocking.strides[0][2] = dst->stridesOf()[isNCHW ? 2 : 1];
        user_dst_md->data.layout_desc.blocking.strides[0][3] = dst->stridesOf()[isNCHW ? 3 : 2];
    }
}

void ConvolutionUtils::getMKLDNNMemoryDescPool3d(
        int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW, int dD, int dH, int dW, int poolingMode, int extraParam0, bool isNCDHW,
        int bS, int iC, int iD, int iH, int iW, int oC, int oD, int oH, int oW,
        const NDArray* src, const NDArray* diff_src, const NDArray* dst, mkldnn::algorithm& algorithm,
        mkldnn::memory::desc* pool_src_md, mkldnn::memory::desc* pool_diff_src_md, mkldnn::memory::desc* pool_dst_md,
        mkldnn::memory::desc* user_src_md, mkldnn::memory::desc* user_diff_src_md, mkldnn::memory::desc* user_dst_md,
        mkldnn::memory::dims& pool_strides, mkldnn::memory::dims& pool_kernel, mkldnn::memory::dims& pool_padding, mkldnn::memory::dims& pool_padding_r) {
    mkldnn::memory::dims pool_src_tz = { bS, iC, iD, iH, iW };
    mkldnn::memory::dims pool_dst_tz = { bS, oC, oD, oH, oW };

    pool_strides = { sD, sH, sW };
    pool_kernel = { kD, kH, kW };
    pool_padding = { pD, pH, pW };
    pool_padding_r = { (oD - 1) * sD - iD + kD - pD,
                       (oH - 1) * sH - iH + kH - pH,
                       (oW - 1) * sW - iW + kW - pW };

    algorithm = poolingMode == 0 ? pooling_max
                                 : extraParam0 == 0 ? pooling_avg_exclude_padding
                                                    : pooling_avg_include_padding;
    auto type = mkldnn::memory::data_type::f32;
    auto format = isNCDHW ? mkldnn::memory::format::ncdhw : mkldnn::memory::format::ndhwc;
    auto supposed_to_be_any_format = mkldnn::memory::format::nCdhw8c; // doesn't work with "any"

    if (src != nullptr && src->getBuffer() != nullptr && pool_src_md != nullptr) {
        *pool_src_md = mkldnn::memory::desc({ pool_src_tz }, type, supposed_to_be_any_format);
        *user_src_md = mkldnn::memory::desc({ pool_src_tz }, type, format);
        user_src_md->data.format = mkldnn_blocked; // overrides "format = isNCDHW ? ncdhw : ndhwc"
        user_src_md->data.layout_desc.blocking.strides[0][0] = src->stridesOf()[isNCDHW ? 0 : 0];
        user_src_md->data.layout_desc.blocking.strides[0][1] = src->stridesOf()[isNCDHW ? 1 : 4];
        user_src_md->data.layout_desc.blocking.strides[0][2] = src->stridesOf()[isNCDHW ? 2 : 1];
        user_src_md->data.layout_desc.blocking.strides[0][3] = src->stridesOf()[isNCDHW ? 3 : 2];
        user_src_md->data.layout_desc.blocking.strides[0][4] = src->stridesOf()[isNCDHW ? 4 : 3];
    }

    if (diff_src != nullptr && diff_src->getBuffer() != nullptr && pool_diff_src_md != nullptr) {
        *pool_diff_src_md = mkldnn::memory::desc({ pool_src_tz }, type, supposed_to_be_any_format);
        *user_diff_src_md = mkldnn::memory::desc({ pool_src_tz }, type, format);
        user_diff_src_md->data.format = mkldnn_blocked; // overrides "format = isNCDHW ? ncdhw : ndhwc"
        user_diff_src_md->data.layout_desc.blocking.strides[0][0] = diff_src->stridesOf()[isNCDHW ? 0 : 0];
        user_diff_src_md->data.layout_desc.blocking.strides[0][1] = diff_src->stridesOf()[isNCDHW ? 1 : 4];
        user_diff_src_md->data.layout_desc.blocking.strides[0][2] = diff_src->stridesOf()[isNCDHW ? 2 : 1];
        user_diff_src_md->data.layout_desc.blocking.strides[0][3] = diff_src->stridesOf()[isNCDHW ? 3 : 2];
        user_diff_src_md->data.layout_desc.blocking.strides[0][4] = diff_src->stridesOf()[isNCDHW ? 4 : 3];
    }

    if (dst != nullptr && dst->getBuffer() != nullptr && pool_dst_md != nullptr) {
        *pool_dst_md = mkldnn::memory::desc({ pool_dst_tz }, type, supposed_to_be_any_format);
        *user_dst_md = mkldnn::memory::desc({ pool_dst_tz }, type, format);
        user_dst_md->data.format = mkldnn_blocked; // overrides "format = isNCDHW ? ncdhw : ndhwc"
        user_dst_md->data.layout_desc.blocking.strides[0][0] = dst->stridesOf()[isNCDHW ? 0 : 0];
        user_dst_md->data.layout_desc.blocking.strides[0][1] = dst->stridesOf()[isNCDHW ? 1 : 4];
        user_dst_md->data.layout_desc.blocking.strides[0][2] = dst->stridesOf()[isNCDHW ? 2 : 1];
        user_dst_md->data.layout_desc.blocking.strides[0][3] = dst->stridesOf()[isNCDHW ? 3 : 2];
        user_dst_md->data.layout_desc.blocking.strides[0][4] = dst->stridesOf()[isNCDHW ? 4 : 3];
    }
}
#endif

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void pooling2d_(nd4j::LaunchContext & block, const NDArray& input, NDArray& output, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0) {
    // input is  [bS, iC, iH, iW]
    // output is [bS, iC, oH, oW]
    T* out = output.bufferAsT<T>();
    T* in  = const_cast<NDArray&>(input).bufferAsT<T>();

    const int kHEff = kH + (kH-1)*(dH-1);
    const int kWEff = kW + (kW-1)*(dW-1);

    const int bS = input.sizeAt(0);
    const int iC = input.sizeAt(1);
    const int iH = input.sizeAt(2);
    const int iW = input.sizeAt(3);
    const int oC = output.sizeAt(1);
    const int oH = output.sizeAt(2);
    const int oW = output.sizeAt(3);

#ifdef HAVE_MKLDNN
    if (poolingMode < 2 && block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported<T, T>()) {
        std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
        if (streams.empty()) {
            streams.push_back(MKLDNNStream("pooling2d"));
        }

        if (streams[0].checkAndReset({&input}, {&output}, {}, {kH, kW, sH, sW, pH, pW, dH, dW, poolingMode, extraParam0})) {
            mkldnn_memory_desc_t empty;
            mkldnn::memory::desc pool_src_md(empty), pool_dst_md(empty);
            mkldnn::memory::desc user_src_md(empty), user_dst_md(empty);
            mkldnn::memory::dims pool_strides, pool_kernel, pool_padding, pool_padding_r;
            mkldnn::algorithm algorithm;

            ConvolutionUtils::getMKLDNNMemoryDescPool2d(kH, kW, sH, sW, pH, pW, dH, dW, poolingMode, extraParam0, true,
                    bS, iC, iH, iW, oC, oH, oW, &input, nullptr, &output, algorithm,
                    &pool_src_md, nullptr, &pool_dst_md, &user_src_md, nullptr, &user_dst_md,
                    pool_strides, pool_kernel, pool_padding, pool_padding_r);

            auto pool_desc = pooling_forward::desc(prop_kind::forward_inference, algorithm, pool_src_md, pool_dst_md,
                    pool_strides, pool_kernel, pool_padding, pool_padding_r, padding_kind::zero);

            auto engine = streams[0].getEngine();
            auto pool_prim_desc = pooling_forward::primitive_desc(pool_desc, engine);
            auto user_src_memory = mkldnn::memory({user_src_md, engine}, const_cast<NDArray&>(input).buffer());
            auto user_dst_memory = mkldnn::memory({user_dst_md, engine}, output.buffer());

            auto pool_src_memory = user_src_memory;
            streams[0].addMemory(user_src_memory);
            if (mkldnn::memory::primitive_desc(pool_prim_desc.src_primitive_desc())
                    != user_src_memory.get_primitive_desc()) {
                pool_src_memory = mkldnn::memory(pool_prim_desc.src_primitive_desc());
                streams[0].addMemory(pool_src_memory);
                streams[0].addOperation(reorder(user_src_memory, pool_src_memory));
            }

            auto pool_dst_memory = user_dst_memory;
            streams[0].addMemory(user_dst_memory);
            if (mkldnn::memory::primitive_desc(pool_prim_desc.dst_primitive_desc())
                    != user_dst_memory.get_primitive_desc()) {
                pool_dst_memory = mkldnn::memory(pool_prim_desc.dst_primitive_desc());
                streams[0].addMemory(pool_dst_memory);
            }

            streams[0].addOperation(pooling_forward(pool_prim_desc, pool_src_memory, pool_dst_memory));

            if (mkldnn::memory::primitive_desc(pool_prim_desc.dst_primitive_desc())
                    != user_dst_memory.get_primitive_desc()) {
                streams[0].addOperation(reorder(pool_dst_memory, user_dst_memory));
            }
        }

        streams[0].submitAndWait();
        return;
    }
#endif
    nd4j_debug("MKL-DNN is not used for pooling2d!\n", 0);

    const Nd4jLong iStride0 = input.stridesOf()[0];
    const Nd4jLong iStride1 = input.stridesOf()[1];
    const Nd4jLong iStride2 = input.stridesOf()[2];
    const Nd4jLong iStride3 = input.stridesOf()[3];    
    const Nd4jLong oStride0 = output.stridesOf()[0];
    const Nd4jLong oStride1 = output.stridesOf()[1];
    const Nd4jLong oStride2 = output.stridesOf()[2];
    const Nd4jLong oStride3 = output.stridesOf()[3];
    
    const Nd4jLong iStep2   = dH*iStride2;
    const Nd4jLong iStep3   = dW*iStride3;    
    const int kProd         = kH*kW;

    Nd4jLong hstart, wstart, hend, wend;
    T *pIn;

    if(poolingMode == 0) {        // max 
        PRAGMA_OMP_PARALLEL_FOR_ARGS(private(pIn, hstart, wstart, hend, wend) collapse(2))
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                                            
                for(int oh = 0; oh < oH; ++oh) {
                    for(int ow = 0; ow < oW; ++ow) {
                        
                        pIn  = in  + b * iStride0 + c * iStride1;
                        
                        hstart = oh * sH - pH;
                        wstart = ow * sW - pW;                        
                        hend = hstart + kHEff;
                        wend = wstart + kWEff;

                        if(hstart < 0)
                            hstart += dH * ((-hstart + dH - 1) / dH); // (Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                        if(wstart < 0)
                            wstart += dW * ((-wstart + dW -1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                        if(hend > iH)
                            hend -= dH * ((hend-iH + dH - 1) / dH); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                        if(wend > iW)
                            wend -= dW * ((wend-iW + dW - 1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                        hstart *= iStride2;
                        hend   *= iStride2;
                        wstart *= iStride3;
                        wend   *= iStride3;

                        T max = -DataTypeUtils::max<T>();

                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2) 
                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3) {
                                T val = pIn[kh + kw];
                                    if (val > max)
                                        max = val;
                            }
                        out[b * oStride0 + c * oStride1 + oh * oStride2 + ow * oStride3] = max;
                    }
                }
            }
        }    
    }
/*************************************************************************/    
    else if(poolingMode == 1) {      // avg
        PRAGMA_OMP_PARALLEL_FOR_ARGS(private(pIn, hstart, wstart, hend, wend) collapse(2))
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                                            
                for(int oh = 0; oh < oH; ++oh) {
                    for(int ow = 0; ow < oW; ++ow) {
                        
                        pIn  = in  + b * iStride0 + c * iStride1;

                        hstart = oh * sH - pH;
                        wstart = ow * sW - pW;
                        hend = hstart + kHEff;
                        wend = wstart + kWEff;

                        if(hstart < 0)
                            hstart += dH * ((-hstart + dH - 1) / dH); // (Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                        if(wstart < 0)
                            wstart += dW * ((-wstart + dW -1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                        if(hend > iH)
                            hend -= dH * ((hend-iH + dH - 1) / dH); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                        if(wend > iW)
                            wend -= dW * ((wend-iW + dW - 1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                        hstart *= iStride2;
                        hend   *= iStride2;
                        wstart *= iStride3;
                        wend   *= iStride3;

                        T sum = static_cast<T>(0.f);

                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2)
                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                sum += pIn[kh + kw];


                        auto oi = b * oStride0 + c * oStride1 + oh * oStride2 + ow * oStride3;

                        if (extraParam0 == 0) {       //Exclude padding
                            int _a = (hend-hstart)/iStep2 + ((hend-hstart) % iStep2 == 0 ? 0 : 1);
                            int _b = (wend-wstart)/iStep3 + ((wend-wstart) % iStep3 == 0 ? 0 : 1);

                            sum /=  _a * _b;   //Accounts for dilation
                        } else if (extraParam0 == 1)  //Include padding
                            sum /= kProd;
                
                        out[oi] = sum;
                    }
                }
            }
        }
    }    
/*************************************************************************/    
    else if(poolingMode == 2) {  // pnorm
        PRAGMA_OMP_PARALLEL_FOR_ARGS(private(pIn, hstart, wstart, hend, wend) collapse(2))
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                                            
                for(int oh = 0; oh < oH; ++oh) {
                    for(int ow = 0; ow < oW; ++ow) {
                        
                        pIn  = in  + b * iStride0 + c * iStride1;

                        hstart = oh * sH - pH;
                        wstart = ow * sW - pW;
                        hend = hstart + kHEff;
                        wend = wstart + kWEff;

                        if(hstart < 0)
                            hstart += dH * ((-hstart + dH - 1) / dH); // (Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                        if(wstart < 0)
                            wstart += dW * ((-wstart + dW -1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                        if(hend > iH)
                            hend -= dH * ((hend-iH + dH - 1) / dH); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                        if(wend > iW)
                            wend -= dW * ((wend-iW + dW - 1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                        hstart *= iStride2;
                        hend   *= iStride2;
                        wstart *= iStride3;
                        wend   *= iStride3;

                        T sum = static_cast<T>(0.f);

                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2) 
                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                sum += nd4j::math::nd4j_pow<T,T,T>(nd4j::math::nd4j_abs<T>(pIn[kh + kw]), extraParam0);
                                
                        sum = nd4j::math::nd4j_pow<T,T,T>(sum, static_cast<T>((T)1.f) / extraParam0);
                                                          
                        out[b * oStride0 + c * oStride1 + oh * oStride2 + ow * oStride3] = sum;
                    }
                }
            }
        }
    }
    else {
        nd4j_printf("ConvolutionUtils::pooling2d: pooling mode argument can take three values only: 0, 1, 2, but got %i instead !\n", poolingMode);
        throw "";
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void pooling3d_(nd4j::LaunchContext & block, const NDArray& input, NDArray& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {
    // input is  [bS, iC, iD, iH, iW]
    // output is [bS, iC, oD, oH, oW]
    T* out = output.bufferAsT<T>();
    T* in  = const_cast<NDArray&>(input).bufferAsT<T>();

    const int kDEff = kD + (kD-1)*(dD-1);
    const int kHEff = kH + (kH-1)*(dH-1);
    const int kWEff = kW + (kW-1)*(dW-1);

    const int bS = input.sizeAt(0);
    const int iC = input.sizeAt(1);
    const int iD = input.sizeAt(2);
    const int iH = input.sizeAt(3);
    const int iW = input.sizeAt(4);
    const int oC = output.sizeAt(1);
    const int oD = output.sizeAt(2);
    const int oH = output.sizeAt(3);
    const int oW = output.sizeAt(4);

#ifdef HAVE_MKLDNN
    if (poolingMode < 2 && block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported<T, T>()) {
        std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
        if (streams.empty()) {
            streams.push_back(MKLDNNStream("pooling3d"));
        }

        if (streams[0].checkAndReset({&input}, {&output}, {}, {kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode, extraParam0})) {
            mkldnn_memory_desc_t empty;
            mkldnn::memory::desc pool_src_md(empty), pool_dst_md(empty);
            mkldnn::memory::desc user_src_md(empty), user_dst_md(empty);
            mkldnn::memory::dims pool_strides, pool_kernel, pool_padding, pool_padding_r;
            mkldnn::algorithm algorithm;

            ConvolutionUtils::getMKLDNNMemoryDescPool3d(kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode, extraParam0, true,
                    bS, iC, iD, iH, iW, oC, oD, oH, oW, &input, nullptr, &output, algorithm,
                    &pool_src_md, nullptr, &pool_dst_md, &user_src_md, nullptr, &user_dst_md,
                    pool_strides, pool_kernel, pool_padding, pool_padding_r);

            auto pool_desc = pooling_forward::desc(prop_kind::forward_inference, algorithm, pool_src_md, pool_dst_md,
                    pool_strides, pool_kernel, pool_padding, pool_padding_r, padding_kind::zero);

            auto engine = streams[0].getEngine();
            auto pool_prim_desc = pooling_forward::primitive_desc(pool_desc, engine);
            auto user_src_memory = mkldnn::memory({user_src_md, engine}, const_cast<NDArray&>(input).buffer());
            auto user_dst_memory = mkldnn::memory({user_dst_md, engine}, output.buffer());

            auto pool_src_memory = user_src_memory;
            streams[0].addMemory(user_src_memory);
            if (mkldnn::memory::primitive_desc(pool_prim_desc.src_primitive_desc())
                    != user_src_memory.get_primitive_desc()) {
                pool_src_memory = mkldnn::memory(pool_prim_desc.src_primitive_desc());
                streams[0].addMemory(pool_src_memory);
                streams[0].addOperation(reorder(user_src_memory, pool_src_memory));
            }

            auto pool_dst_memory = user_dst_memory;
            streams[0].addMemory(user_dst_memory);
            if (mkldnn::memory::primitive_desc(pool_prim_desc.dst_primitive_desc())
                    != user_dst_memory.get_primitive_desc()) {
                pool_dst_memory = mkldnn::memory(pool_prim_desc.dst_primitive_desc());
                streams[0].addMemory(pool_dst_memory);
            }

            streams[0].addOperation(pooling_forward(pool_prim_desc, pool_src_memory, pool_dst_memory));

            if (mkldnn::memory::primitive_desc(pool_prim_desc.dst_primitive_desc())
                    != user_dst_memory.get_primitive_desc()) {
                streams[0].addOperation(reorder(pool_dst_memory, user_dst_memory));
            }
        }

        streams[0].submitAndWait();
        return;
    }
#endif
    nd4j_debug("MKL-DNN is not used for pooling3d!\n", 0);

    const Nd4jLong iStride0 = input.stridesOf()[0];
    const Nd4jLong iStride1 = input.stridesOf()[1];
    const Nd4jLong iStride2 = input.stridesOf()[2];
    const Nd4jLong iStride3 = input.stridesOf()[3];
    const Nd4jLong iStride4 = input.stridesOf()[4];
    const Nd4jLong oStride0 = output.stridesOf()[0];
    const Nd4jLong oStride1 = output.stridesOf()[1];
    const Nd4jLong oStride2 = output.stridesOf()[2];
    const Nd4jLong oStride3 = output.stridesOf()[3];
    const Nd4jLong oStride4 = output.stridesOf()[4];
    const Nd4jLong iStep2   = dD*iStride2;
    const Nd4jLong iStep3   = dH*iStride3;
    const Nd4jLong iStep4   = dW*iStride4;
    const int kProd         = kD*kH*kW;
    const T iStep2Inv = 1./iStep2;
    const T iStep3Inv = 1./iStep3;
    const T iStep4Inv = 1./iStep4;

    Nd4jLong dstart, hstart, wstart, dend, hend, wend;
    T sum, *pIn;

    if(poolingMode == 0) {        // max 
        PRAGMA_OMP_PARALLEL_FOR_ARGS(private(pIn, sum, dstart, hstart, wstart, dend, hend, wend))
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                            
                for(int od = 0; od < oD; ++od) {
                    for(int oh = 0; oh < oH; ++oh) {
                        for(int ow = 0; ow < oW; ++ow) {
                        
                            pIn  = in  + b * iStride0 + c * iStride1;

                            dstart = od * sD - pD;
                            hstart = oh * sH - pH;
                            wstart = ow * sW - pW;
                            dend = dstart + kDEff;
                            hend = hstart + kHEff;
                            wend = wstart + kWEff;

                            if(dstart < 0)
                                dstart += dD * ((-dstart + dD - 1) / dD);
                            if(hstart < 0)
                                hstart += dH * ((-hstart + dH - 1) / dH);
                            if(wstart < 0)
                                wstart += dW * ((-wstart + dW - 1) / dW);
                            if(dend > iD)
                                dend -= dD * ((dend-iD + dD - 1) / dD);
                            if(hend > iH)
                                hend -= dH * ((hend-iH + dH - 1) / dH);
                            if(wend > iW)
                                wend -= dW * ((wend-iW + dW - 1) / dW);

                            dstart *= iStride2;
                            dend   *= iStride2;
                            hstart *= iStride3;
                            hend   *= iStride3;
                            wstart *= iStride4;
                            wend   *= iStride4;

                            sum = -DataTypeUtils::max<T>();
                                            
                            for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) 
                                for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) 
                                    for (Nd4jLong kw = wstart; kw < wend; kw += iStep4) {
                                        T val = pIn[kd + kh + kw];
                                            if (val > sum)
                                            sum = val;
                                    }
                            out[b * oStride0 + c * oStride1 + od * oStride2 + oh * oStride3 + ow * oStride4] = sum;
                        }
                    }
                }
            }
        }
    }  
/*************************************************************************/    
    else if(poolingMode == 1) {     // avg
        PRAGMA_OMP_PARALLEL_FOR_ARGS(private(pIn, sum, dstart, hstart, wstart, dend, hend, wend))
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                            
                for(int od = 0; od < oD; ++od) {
                    for(int oh = 0; oh < oH; ++oh) {
                        for(int ow = 0; ow < oW; ++ow) {
                        
                            pIn  = in  + b * iStride0 + c * iStride1;

                            dstart = od * sD - pD;
                            hstart = oh * sH - pH;
                            wstart = ow * sW - pW;
                            dend = dstart + kDEff;
                            hend = hstart + kHEff;
                            wend = wstart + kWEff;

                            if(dstart < 0)
                                dstart += dD * ((-dstart + dD - 1) / dD);
                            if(hstart < 0)
                                hstart += dH * ((-hstart + dH - 1) / dH);
                            if(wstart < 0)
                                wstart += dW * ((-wstart + dW - 1) / dW);
                            if(dend > iD)
                                dend -= dD * ((dend-iD + dD - 1) / dD);
                            if(hend > iH)
                                hend -= dH * ((hend-iH + dH - 1) / dH);
                            if(wend > iW)
                                wend -= dW * ((wend-iW + dW - 1) / dW);

                            dstart *= iStride2;
                            dend   *= iStride2;
                            hstart *= iStride3;
                            hend   *= iStride3;
                            wstart *= iStride4;
                            wend   *= iStride4;

                            sum = static_cast<T>(0.);
                                            
                            for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) 
                                for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) 
                                    for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                        sum += pIn[kd + kh + kw];
                                
                            if ((int) extraParam0 == 0)         //Exclude padding
                                sum /= static_cast<T>(nd4j::math::nd4j_ceil<double,T>(static_cast<double>(dend-dstart) / static_cast<double>(iStep2))) * static_cast<T>(nd4j::math::nd4j_ceil<double,T>(static_cast<double>(hend-hstart) / static_cast<double>(iStep3))) * static_cast<double>(nd4j::math::nd4j_ceil<double,double>(static_cast<double>(wend-wstart) / static_cast<double>(iStep4)));   //Accounts for dilation
                            else if ((int) extraParam0 == 1)    //Include padding
                                sum /= kProd;
                    
                            out[b * oStride0 + c * oStride1 + od * oStride2 + oh * oStride3 + ow * oStride4] = sum;
                        }
                    }
                }
            }
        }
    }
/*************************************************************************/    
    else if(poolingMode == 2) {  // pnorm
        PRAGMA_OMP_PARALLEL_FOR_ARGS(private(pIn, sum, dstart, hstart, wstart, dend, hend, wend))
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                            
                for(int od = 0; od < oD; ++od) {
                    for(int oh = 0; oh < oH; ++oh) {
                        for(int ow = 0; ow < oW; ++ow) {
                        
                            pIn  = in  + b * iStride0 + c * iStride1;

                            dstart = od * sD - pD;
                            hstart = oh * sH - pH;
                            wstart = ow * sW - pW;
                            dend = dstart + kDEff;
                            hend = hstart + kHEff;
                            wend = wstart + kWEff;

                            if(dstart < 0)
                                dstart += dD * ((-dstart + dD - 1) / dD);
                            if(hstart < 0)
                                hstart += dH * ((-hstart + dH - 1) / dH);
                            if(wstart < 0)
                                wstart += dW * ((-wstart + dW - 1) / dW);
                            if(dend > iD)
                                dend -= dD * ((dend-iD + dD - 1) / dD);
                            if(hend > iH)
                                hend -= dH * ((hend-iH + dH - 1) / dH);
                            if(wend > iW)
                                wend -= dW * ((wend-iW + dW - 1) / dW);

                            dstart *= iStride2;
                            dend   *= iStride2;
                            hstart *= iStride3;
                            hend   *= iStride3;
                            wstart *= iStride4;
                            wend   *= iStride4;

                            sum = static_cast<T>(0.);
                                            
                            for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) 
                                for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) 
                                    for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                        sum += nd4j::math::nd4j_pow<T,T,T>(nd4j::math::nd4j_abs<T>(pIn[kd + kh + kw]), extraParam0);
                                
                            sum = nd4j::math::nd4j_pow<T,T,T>(sum, (T) 1.f / extraParam0);
                                                          
                            out[b * oStride0 + c * oStride1 + od * oStride2 + oh * oStride3 + ow * oStride4] = sum;
                        }
                    }
                }
            }
        }
    }
    else {
        nd4j_printf("ConvolutionUtils::pooling3d: pooling mode argument can take three values only: 0, 1, 2, but got %i instead !\n", poolingMode);
        throw "";
    }
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
static void pooling2dBP_(nd4j::LaunchContext & block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0) {
    // input [bS, iC, iH, iW]
    // gradI [bS, iC, iH, iW] -> gradI is output in this function
    // gradO [bS, iC, oH, oW]    

    T* in = const_cast<NDArray&>(input).bufferAsT<T>();
    T* gO = const_cast<NDArray&>(gradO).bufferAsT<T>();
    T* gI = gradI.bufferAsT<T>();

    // initial zeroing of gradI
    const Nd4jLong gradIEWS = gradI.ews();
    const Nd4jLong gradILen = gradI.lengthOf();
    if(gradIEWS == 1)
        memset(gI, 0, gradILen * sizeof(T));
    else if (gradIEWS > 1) {
        for (Nd4jLong i = 0; i < gradILen * gradIEWS; i += gradIEWS)
            gI[i] = static_cast<T>(0.f);
    }
    else {
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (Nd4jLong i = 0; i < gradILen; i++) 
            gI[shape::getIndexOffset(i, gradI.getShapeInfo(), gradILen)] = static_cast<T>(0.f);        
    }

    const int kHEff = kH + (kH-1)*(dH-1);
    const int kWEff = kW + (kW-1)*(dW-1);

    const int bS = gradI.sizeAt(0);
    const int iC = gradI.sizeAt(1);
    const int iH = gradI.sizeAt(2);
    const int iW = gradI.sizeAt(3);
    const int oC = gradO.sizeAt(1);
    const int oH = gradO.sizeAt(2);
    const int oW = gradO.sizeAt(3);

#ifdef HAVE_MKLDNN
    if (poolingMode < 2 && block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported<T, T>()) {
        std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
        if (streams.empty()) {
            streams.push_back(MKLDNNStream("pooling2d_bp"));
        }

        if (streams[0].checkAndReset({&input, &gradO}, {&gradI}, {}, {kH, kW, sH, sW, pH, pW, dH, dW, poolingMode, extraParam0})) {
            mkldnn_memory_desc_t empty;
            mkldnn::memory::desc pool_src_md(empty), pool_diff_src_md(empty), pool_dst_md(empty);
            mkldnn::memory::desc user_src_md(empty), user_diff_src_md(empty), user_dst_md(empty);
            mkldnn::memory::dims pool_strides, pool_kernel, pool_padding, pool_padding_r;
            mkldnn::algorithm algorithm;

            ConvolutionUtils::getMKLDNNMemoryDescPool2d(kH, kW, sH, sW, pH, pW, dH, dW, poolingMode, extraParam0, true,
                    bS, iC, iH, iW, oC, oH, oW, &input, &gradI, &gradO, algorithm,
                    &pool_src_md, &pool_diff_src_md, &pool_dst_md, &user_src_md, &user_diff_src_md, &user_dst_md,
                    pool_strides, pool_kernel, pool_padding, pool_padding_r);

            // input is sometimes null, so we can't rely on pool_src_md being valid
            auto pool_desc = pooling_forward::desc(prop_kind::forward, algorithm,
                    const_cast<NDArray&>(input).buffer() != nullptr ? pool_src_md : pool_diff_src_md,
                    pool_dst_md, pool_strides, pool_kernel, pool_padding, pool_padding_r, padding_kind::zero);

            auto engine = streams[0].getEngine();
            auto pool_prim_desc = pooling_forward::primitive_desc(pool_desc, engine);

            auto poolB_desc = pooling_backward::desc(algorithm, pool_diff_src_md, pool_dst_md,
                    pool_strides, pool_kernel, pool_padding, pool_padding_r, padding_kind::zero);

            auto poolB_prim_desc = pooling_backward::primitive_desc(poolB_desc, engine, pool_prim_desc);
            auto userB_src_memory = mkldnn::memory({user_src_md, engine}, gradI.buffer());
            auto userB_dst_memory = mkldnn::memory({user_dst_md, engine}, const_cast<NDArray&>(gradO).buffer());

            auto poolB_src_memory = userB_src_memory;
            streams[0].addMemory(userB_src_memory);
            if (mkldnn::memory::primitive_desc(poolB_prim_desc.diff_src_primitive_desc())
                    != userB_src_memory.get_primitive_desc()) {
                poolB_src_memory = mkldnn::memory(poolB_prim_desc.diff_src_primitive_desc());
                streams[0].addMemory(poolB_src_memory);
            }

            auto poolB_dst_memory = userB_dst_memory;
            streams[0].addMemory(userB_dst_memory);
            if (mkldnn::memory::primitive_desc(poolB_prim_desc.diff_dst_primitive_desc())
                    != userB_dst_memory.get_primitive_desc()) {
                poolB_dst_memory = mkldnn::memory(poolB_prim_desc.diff_dst_primitive_desc());
                streams[0].addMemory(poolB_dst_memory);
                streams[0].addOperation(reorder(userB_dst_memory, poolB_dst_memory));
            }

            if (algorithm == mkldnn::pooling_max) {
                auto user_src_memory = mkldnn::memory({user_src_md, engine}, const_cast<NDArray&>(input).buffer());

                auto pool_src_memory = user_src_memory;
                streams[0].addMemory(user_src_memory);
                if (mkldnn::memory::primitive_desc(pool_prim_desc.src_primitive_desc())
                        != user_src_memory.get_primitive_desc()) {
                    pool_src_memory = mkldnn::memory(pool_prim_desc.src_primitive_desc());
                    streams[0].addMemory(pool_src_memory);
                    streams[0].addOperation(reorder(user_src_memory, pool_src_memory));
                }

                auto pool_dst_memory = mkldnn::memory(pool_prim_desc.dst_primitive_desc());
                streams[0].addMemory(pool_dst_memory);

                auto pool_workspace_memory = mkldnn::memory(pool_prim_desc.workspace_primitive_desc());
                streams[0].addMemory(pool_workspace_memory);

                streams[0].addOperation(pooling_forward(pool_prim_desc, pool_src_memory, pool_dst_memory, pool_workspace_memory));
                streams[0].addOperation(pooling_backward(poolB_prim_desc, poolB_dst_memory, pool_workspace_memory, poolB_src_memory));
            } else {
                streams[0].addOperation(pooling_backward(poolB_prim_desc, poolB_dst_memory, poolB_src_memory));
            }

            if (mkldnn::memory::primitive_desc(poolB_prim_desc.diff_src_primitive_desc())
                    != userB_src_memory.get_primitive_desc()) {
                streams[0].addOperation(reorder(poolB_src_memory, userB_src_memory));
            }
        }

        streams[0].submitAndWait();
        return;
    }
#endif
    nd4j_debug("MKL-DNN is not used for pooling2d_bp!\n", 0);

    const Nd4jLong iStride0 = gradI.stridesOf()[0];
    const Nd4jLong iStride1 = gradI.stridesOf()[1];
    const Nd4jLong iStride2 = gradI.stridesOf()[2];
    const Nd4jLong iStride3 = gradI.stridesOf()[3];
    const Nd4jLong oStride0 = gradO.stridesOf()[0];
    const Nd4jLong oStride1 = gradO.stridesOf()[1];
    const Nd4jLong oStride2 = gradO.stridesOf()[2];
    const Nd4jLong oStride3 = gradO.stridesOf()[3];
    const Nd4jLong iStep2   = dH*iStride2;
    const Nd4jLong iStep3   = dW*iStride3;
    const int      kProd    = kH*kW;
    const T iStep2Inv = 1./iStep2;
    const T iStep3Inv = 1./iStep3;

    Nd4jLong hstart, wstart,hend, wend, maxKH, maxKW;
    T sum, valO, *pIn, *pgI;

    if(poolingMode == 0) {        // max 
        PRAGMA_OMP_PARALLEL_FOR_ARGS(private(pIn, valO, sum, hstart, wstart, hend, wend, maxKH, maxKW))
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                            
                for(int oh = 0; oh < oH; ++oh) {
                    for(int ow = 0; ow < oW; ++ow) {
                    
                        pIn = in + b * iStride0 + c * iStride1;

                        hstart = oh * sH - pH;
                        wstart = ow * sW - pW;
                        hend = hstart + kHEff;
                        wend = wstart + kWEff;

                        if(hstart < 0)
                            hstart += dH * ((-hstart + dH - 1) / dH); // (Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                        if(wstart < 0)
                            wstart += dW * ((-wstart + dW -1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                        if(hend > iH)
                            hend -= dH * ((hend-iH + dH - 1) / dH); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                        if(wend > iW)
                            wend -= dW * ((wend-iW + dW - 1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                        hstart *= iStride2;
                        hend   *= iStride2;
                        wstart *= iStride3;
                        wend   *= iStride3;

                        sum = -DataTypeUtils::max<T>();
                        valO = gO[b*oStride0 + c*oStride1 + oh*oStride2 + ow*oStride3];

                        // we set these to default values
                        maxKH = hstart;
                        maxKW = wstart;

                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2)
                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3) {
                                T valIn = pIn[kh + kw];
                                if (valIn > sum) {
                                    sum = valIn;
                                    maxKH = kh;
                                    maxKW = kw;
                                }
                            }
                        gI[pIn - in + maxKH + maxKW] += valO;
                    }
                }
            }
        }
    }  
/*************************************************************************/    
    else if(poolingMode == 1) {     // avg        
        PRAGMA_OMP_PARALLEL_FOR_ARGS(private(pgI, valO, hstart, wstart, hend, wend))
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                            
                for(int oh = 0; oh < oH; ++oh) {
                    for(int ow = 0; ow < oW; ++ow) {
                        
                        pgI  = gI + b * iStride0 + c * iStride1;

                        hstart = oh * sH - pH;
                        wstart = ow * sW - pW;
                        hend = hstart + kHEff;
                        wend = wstart + kWEff;

                        if(hstart < 0)
                            hstart += dH * ((-hstart + dH - 1) / dH); // (Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                        if(wstart < 0)
                            wstart += dW * ((-wstart + dW -1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                        if(hend > iH)
                            hend -= dH * ((hend-iH + dH - 1) / dH); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                        if(wend > iW)
                            wend -= dW * ((wend-iW + dW - 1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                        hstart *= iStride2;
                        hend   *= iStride2;
                        wstart *= iStride3;
                        wend   *= iStride3;

                        valO = gO[b*oStride0 + c*oStride1 + oh*oStride2 + ow*oStride3];
                                            
                        if ((int) extraParam0 == 0)         //Exclude padding                            
                            valO /= static_cast<T>(nd4j::math::nd4j_ceil<double,T>(static_cast<double>(hend-hstart) / static_cast<double>(iStep2))) * static_cast<T>(nd4j::math::nd4j_ceil<double,T>(static_cast<double>(wend-wstart) / static_cast<double>(iStep3)));   //Accounts for dilation
                        else if ((int) extraParam0 == 1)    //Include padding
                            valO /= kProd;

                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2) 
                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                pgI[kh + kw] += valO;
                    }
                }
            }
        }
    }
/*************************************************************************/    
    else if(poolingMode == 2) {  // pnorm
        PRAGMA_OMP_PARALLEL_FOR_ARGS(private(pIn, valO, pgI, sum, hstart, wstart, hend, wend))
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                            
                for(int oh = 0; oh < oH; ++oh) {
                    for(int ow = 0; ow < oW; ++ow) {
                        
                        pIn  = in + b * iStride0 + c * iStride1;
                        pgI  = gI + (pIn - in);

                        hstart = oh * sH - pH;
                        wstart = ow * sW - pW;
                        hend = hstart + kHEff;
                        wend = wstart + kWEff;

                        if(hstart < 0)
                            hstart += dH * ((-hstart + dH - 1) / dH); // (Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                        if(wstart < 0)
                            wstart += dW * ((-wstart + dW -1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                        if(hend > iH)
                            hend -= dH * ((hend-iH + dH - 1) / dH); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                        if(wend > iW)
                            wend -= dW * ((wend-iW + dW - 1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(wend-iW) / static_cast<T>(dW));


                        hstart *= iStride2;
                        hend   *= iStride2;
                        wstart *= iStride3;
                        wend   *= iStride3;

                        sum = static_cast<T>(0.f);
                        valO = gO[b*oStride0 + c*oStride1 + oh*oStride2 + ow*oStride3];
                                            
                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2) 
                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                sum += nd4j::math::nd4j_pow<T,T,T>(nd4j::math::nd4j_abs<T>(pIn[kh + kw]), extraParam0);
                                
                        valO *= nd4j::math::nd4j_pow<T,T,T>(sum, ((T)1. - extraParam0) / extraParam0);

                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2) 
                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                pgI[kh + kw] += valO * nd4j::math::nd4j_pow<T,T,T>(nd4j::math::nd4j_abs<T>(pIn[kh + kw]), extraParam0 - 1.f);
                    }
                }
            }
        }
    }
    else {
        nd4j_printf("ConvolutionUtils::pooling2dBP: pooling mode argument can take three values only: 0, 1, 2, but got %i instead !\n", poolingMode);
        throw "";
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void pooling3dBP_(nd4j::LaunchContext & block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {
    // input [bS, iC, iD, iH, iW]
    // gradI [bS, iC, iD, iH, iW] -> gradI is output in this function
    // gradO [bS, iC, oD, oH, oW]
    
    T* in = const_cast<NDArray&>(input).bufferAsT<T>();
    T* gO = const_cast<NDArray&>(gradO).bufferAsT<T>();
    T* gI = gradI.bufferAsT<T>();

    // initial zeroing of gradI
    const Nd4jLong gradIEWS = gradI.ews();
    const Nd4jLong gradILen = gradI.lengthOf();
    if(gradIEWS == 1) {
        memset(gI, 0, gradILen * sizeof(T));
    }
    else if (gradIEWS > 1) {
        PRAGMA_OMP_PARALLEL_FOR
        for (Nd4jLong i = 0; i < gradILen * gradIEWS; i += gradIEWS)
            gI[i] = static_cast<T>(0.f);
    }
    else {
        PRAGMA_OMP_PARALLEL_FOR
        for (Nd4jLong i = 0; i < gradILen; i++)             
            gI[shape::getIndexOffset(i, gradI.getShapeInfo(), gradILen)] = static_cast<T>(0.f);        
    }

    const int kDEff = kD + (kD-1)*(dD-1);
    const int kHEff = kH + (kH-1)*(dH-1);
    const int kWEff = kW + (kW-1)*(dW-1);

    const int bS = gradI.sizeAt(0);
    const int iC = gradI.sizeAt(1);
    const int iD = gradI.sizeAt(2);
    const int iH = gradI.sizeAt(3);
    const int iW = gradI.sizeAt(4);
    const int oC = gradO.sizeAt(1);
    const int oD = gradO.sizeAt(2);
    const int oH = gradO.sizeAt(3);
    const int oW = gradO.sizeAt(4);

#ifdef HAVE_MKLDNN
    if (poolingMode < 2 && block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported<T, T>()) {
        std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
        if (streams.empty()) {
            streams.push_back(MKLDNNStream("pooling3d_bp"));
        }

        if (streams[0].checkAndReset({&input, &gradO}, {&gradI}, {}, {kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode, extraParam0})) {
            mkldnn_memory_desc_t empty;
            mkldnn::memory::desc pool_src_md(empty), pool_diff_src_md(empty), pool_dst_md(empty);
            mkldnn::memory::desc user_src_md(empty), user_diff_src_md(empty), user_dst_md(empty);
            mkldnn::memory::dims pool_strides, pool_kernel, pool_padding, pool_padding_r;
            mkldnn::algorithm algorithm;

            ConvolutionUtils::getMKLDNNMemoryDescPool3d(kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode, extraParam0, true,
                    bS, iC, iD, iH, iW, oC, oD, oH, oW, &input, &gradI, &gradO, algorithm,
                    &pool_src_md, &pool_diff_src_md, &pool_dst_md, &user_src_md, &user_diff_src_md, &user_dst_md,
                    pool_strides, pool_kernel, pool_padding, pool_padding_r);

            // input is sometimes null, so we can't rely on pool_src_md being valid
            if (const_cast<NDArray&>(input).buffer() == nullptr) {
                pool_src_md = pool_diff_src_md;
                user_src_md = user_diff_src_md;
            }
            auto pool_desc = pooling_forward::desc(prop_kind::forward, algorithm, pool_src_md,
                    pool_dst_md, pool_strides, pool_kernel, pool_padding, pool_padding_r, padding_kind::zero);

            auto engine = streams[0].getEngine();
            auto pool_prim_desc = pooling_forward::primitive_desc(pool_desc, engine);

            auto poolB_desc = pooling_backward::desc(algorithm, pool_diff_src_md, pool_dst_md,
                    pool_strides, pool_kernel, pool_padding, pool_padding_r, padding_kind::zero);

            auto poolB_prim_desc = pooling_backward::primitive_desc(poolB_desc, engine, pool_prim_desc);
            auto userB_src_memory = mkldnn::memory({user_diff_src_md, engine}, gradI.buffer());
            auto userB_dst_memory = mkldnn::memory({user_dst_md, engine}, const_cast<NDArray&>(gradO).buffer());

            auto poolB_src_memory = userB_src_memory;
            streams[0].addMemory(userB_src_memory);
            if (mkldnn::memory::primitive_desc(poolB_prim_desc.diff_src_primitive_desc())
                    != userB_src_memory.get_primitive_desc()) {
                poolB_src_memory = mkldnn::memory(poolB_prim_desc.diff_src_primitive_desc());
                streams[0].addMemory(poolB_src_memory);
            }

            auto poolB_dst_memory = userB_dst_memory;
            streams[0].addMemory(userB_dst_memory);
            if (mkldnn::memory::primitive_desc(poolB_prim_desc.diff_dst_primitive_desc())
                    != userB_dst_memory.get_primitive_desc()) {
                poolB_dst_memory = mkldnn::memory(poolB_prim_desc.diff_dst_primitive_desc());
                streams[0].addMemory(poolB_dst_memory);
                streams[0].addOperation(reorder(userB_dst_memory, poolB_dst_memory));
            }

            if (algorithm == mkldnn::pooling_max) {
                auto user_src_memory = mkldnn::memory({user_src_md, engine}, const_cast<NDArray&>(input).buffer());

                auto pool_src_memory = user_src_memory;
                streams[0].addMemory(user_src_memory);
                if (mkldnn::memory::primitive_desc(pool_prim_desc.src_primitive_desc())
                        != user_src_memory.get_primitive_desc()) {
                    pool_src_memory = mkldnn::memory(pool_prim_desc.src_primitive_desc());
                    streams[0].addMemory(pool_src_memory);
                    streams[0].addOperation(reorder(user_src_memory, pool_src_memory));
                }

                auto pool_dst_memory = mkldnn::memory(pool_prim_desc.dst_primitive_desc());
                streams[0].addMemory(pool_dst_memory);

                auto pool_workspace_memory = mkldnn::memory(pool_prim_desc.workspace_primitive_desc());
                streams[0].addMemory(pool_workspace_memory);

                streams[0].addOperation(pooling_forward(pool_prim_desc, pool_src_memory, pool_dst_memory, pool_workspace_memory));
                streams[0].addOperation(pooling_backward(poolB_prim_desc, poolB_dst_memory, pool_workspace_memory, poolB_src_memory));
            } else {
                streams[0].addOperation(pooling_backward(poolB_prim_desc, poolB_dst_memory, poolB_src_memory));
            }

            if (mkldnn::memory::primitive_desc(poolB_prim_desc.diff_src_primitive_desc())
                    != userB_src_memory.get_primitive_desc()) {
                streams[0].addOperation(reorder(poolB_src_memory, userB_src_memory));
            }
        }

        streams[0].submitAndWait();
        return;
    }
#endif
    nd4j_debug("MKL-DNN is not used for pooling3d_bp!\n", 0);

    const Nd4jLong iStride0 = gradI.stridesOf()[0];
    const Nd4jLong iStride1 = gradI.stridesOf()[1];
    const Nd4jLong iStride2 = gradI.stridesOf()[2];
    const Nd4jLong iStride3 = gradI.stridesOf()[3];
    const Nd4jLong iStride4 = gradI.stridesOf()[4];
    const Nd4jLong oStride0 = gradO.stridesOf()[0];
    const Nd4jLong oStride1 = gradO.stridesOf()[1];
    const Nd4jLong oStride2 = gradO.stridesOf()[2];
    const Nd4jLong oStride3 = gradO.stridesOf()[3];
    const Nd4jLong oStride4 = gradO.stridesOf()[4];
    const Nd4jLong iStep2   = dD*iStride2;
    const Nd4jLong iStep3   = dH*iStride3;
    const Nd4jLong iStep4   = dW*iStride4;
    const int      kProd    = kD*kH*kW;
    const T iStep2Inv = 1./iStep2;
    const T iStep3Inv = 1./iStep3;
    const T iStep4Inv = 1./iStep4;

    Nd4jLong dstart, hstart, wstart, dend, hend, wend, maxKD, maxKH, maxKW;
    T sum, valO, *pIn, *pgI;

    if(poolingMode == 0) {        // max 
        PRAGMA_OMP_PARALLEL_FOR_ARGS(private(pIn, valO, sum, dstart, hstart, wstart, dend, hend, wend, maxKD, maxKH, maxKW))
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                            
                for(int od = 0; od < oD; ++od) {
                    for(int oh = 0; oh < oH; ++oh) {
                        for(int ow = 0; ow < oW; ++ow) {
                                                    
                            pIn = in + b * iStride0 + c * iStride1;

                            dstart = od * sD - pD;
                            hstart = oh * sH - pH;
                            wstart = ow * sW - pW;
                            dend = dstart + kDEff;
                            hend = hstart + kHEff;
                            wend = wstart + kWEff;

                            if(dstart < 0)
                                dstart += dD * ((-dstart + dD - 1) / dD);
                            if(hstart < 0)
                                hstart += dH * ((-hstart + dH - 1) / dH);
                            if(wstart < 0)
                                wstart += dW * ((-wstart + dW - 1) / dW);
                            if(dend > iD)
                                dend -= dD * ((dend-iD + dD - 1) / dD);
                            if(hend > iH)
                                hend -= dH * ((hend-iH + dH - 1) / dH);
                            if(wend > iW)
                                wend -= dW * ((wend-iW + dW - 1) / dW);

                            dstart *= iStride2;
                            dend   *= iStride2;
                            hstart *= iStride3;
                            hend   *= iStride3;
                            wstart *= iStride4;
                            wend   *= iStride4;

                            maxKD = dstart;
                            maxKH = hstart;
                            maxKW = wstart;

                            sum = -DataTypeUtils::max<T>();
                            valO = gO[b*oStride0 + c*oStride1+ od*oStride2 + oh*oStride3 + ow*oStride4];
                            
                            for (Nd4jLong kd = dstart; kd < dend; kd += iStep2)
                                for (Nd4jLong kh = hstart; kh < hend; kh += iStep3)
                                    for (Nd4jLong kw = wstart; kw < wend; kw += iStep4) {
                                        T valIn = pIn[kd + kh + kw];
                                        if (valIn > sum) {
                                            sum = valIn;
                                            maxKD = kd;
                                            maxKH = kh;
                                            maxKW = kw;
                                        }
                                    }
                            gI[pIn - in + maxKD + maxKH + maxKW] += valO;
                        }
                    }
                }
            }
        }
    }  
/*************************************************************************/    
    else if(poolingMode == 1) {     // avg        
        PRAGMA_OMP_PARALLEL_FOR_ARGS(private(pgI, valO, dstart, hstart, wstart, dend, hend, wend))
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                            
                for(int od = 0; od < oD; ++od) {
                    for(int oh = 0; oh < oH; ++oh) {
                        for(int ow = 0; ow < oW; ++ow) {
                        
                            pgI  = gI + b * iStride0 + c * iStride1;

                            dstart = od * sD - pD;
                            hstart = oh * sH - pH;
                            wstart = ow * sW - pW;
                            dend = dstart + kDEff;
                            hend = hstart + kHEff;
                            wend = wstart + kWEff;

                            if(dstart < 0)
                                dstart += dD * ((-dstart + dD - 1) / dD);
                            if(hstart < 0)
                                hstart += dH * ((-hstart + dH - 1) / dH);
                            if(wstart < 0)
                                wstart += dW * ((-wstart + dW - 1) / dW);
                            if(dend > iD)
                                dend -= dD * ((dend-iD + dD - 1) / dD);
                            if(hend > iH)
                                hend -= dH * ((hend-iH + dH - 1) / dH);
                            if(wend > iW)
                                wend -= dW * ((wend-iW + dW - 1) / dW);

                            dstart *= iStride2;
                            dend   *= iStride2;
                            hstart *= iStride3;
                            hend   *= iStride3;
                            wstart *= iStride4;
                            wend   *= iStride4;

                            valO = gO[b*oStride0 + c*oStride1+ od*oStride2 + oh*oStride3 + ow*oStride4];
                                            
                            if ((int) extraParam0 == 0)         //Exclude padding
                                valO /= static_cast<T>(nd4j::math::nd4j_ceil<double,T>(static_cast<double>(dend-dstart) / static_cast<double>(iStep2))) * static_cast<T>(nd4j::math::nd4j_ceil<double,T>(static_cast<double>(hend-hstart) / static_cast<double>(iStep3))) * static_cast<double>(nd4j::math::nd4j_ceil<double,double>(static_cast<double>(wend-wstart) / static_cast<double>(iStep4)));   //Accounts for dilation
                            else if ((int) extraParam0 == 1)    //Include padding
                                valO /= kProd;

                            for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) 
                                for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) 
                                    for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                        pgI[kd + kh + kw] += valO;
                        }
                    }
                }
            }
        }
    }
/*************************************************************************/    
    else if(poolingMode == 2) {  // pnorm
        PRAGMA_OMP_PARALLEL_FOR_ARGS(private(pIn, pgI, valO, sum, dstart, hstart, wstart, dend, hend, wend))
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                            
                for(int od = 0; od < oD; ++od) {
                    for(int oh = 0; oh < oH; ++oh) {
                        for(int ow = 0; ow < oW; ++ow) {
                        
                            pIn  = in + b * iStride0 + c * iStride1;
                            pgI  = gI + (pIn - in);

                            dstart = od * sD - pD;
                            hstart = oh * sH - pH;
                            wstart = ow * sW - pW;
                            dend = dstart + kDEff;
                            hend = hstart + kHEff; 
                            wend = wstart + kWEff;

                            if(dstart < 0)
                                dstart += dD * ((-dstart + dD - 1) / dD);
                            if(hstart < 0)
                                hstart += dH * ((-hstart + dH - 1) / dH);
                            if(wstart < 0)
                                wstart += dW * ((-wstart + dW - 1) / dW);
                            if(dend > iD)
                                dend -= dD * ((dend-iD + dD - 1) / dD);
                            if(hend > iH)
                                hend -= dH * ((hend-iH + dH - 1) / dH);
                            if(wend > iW)
                                wend -= dW * ((wend-iW + dW - 1) / dW);

                            dstart *= iStride2;
                            dend   *= iStride2;
                            hstart *= iStride3;
                            hend   *= iStride3;
                            wstart *= iStride4;
                            wend   *= iStride4;

                            sum = static_cast<T>(0.);
                            valO = gO[b*oStride0 + c*oStride1+ od*oStride2 + oh*oStride3 + ow*oStride4];

                            for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) 
                                for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) 
                                    for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                        sum += nd4j::math::nd4j_pow<T,T,T>(nd4j::math::nd4j_abs<T>(pIn[kd + kh + kw]), extraParam0);

                            valO *= nd4j::math::nd4j_pow<T,T,T>(sum, ((T)1.f - extraParam0) / extraParam0);

                            for (Nd4jLong kd = dstart; kd < dend; kd += iStep2) 
                                for (Nd4jLong kh = hstart; kh < hend; kh += iStep3) 
                                    for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                        pgI[kd + kh + kw] += valO * nd4j::math::nd4j_pow<T,T,T>(nd4j::math::nd4j_abs<T>(pIn[kd + kh + kw]), extraParam0 - 1.f);
                        }
                    }
                }
            }
        }
    }
    else {
        nd4j_printf("ConvolutionUtils::pooling3dBP: pooling mode argument can take three values only: 0, 1, 2, but got %i instead !\n", poolingMode);
        throw "";
    }
}




void ConvolutionUtils::conv2d(nd4j::LaunchContext & block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {
    BUILD_DOUBLE_SELECTOR(input->dataType(), output->dataType(), conv2d_, (block, input, weights, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW), LIBND4J_TYPES, FLOAT_TYPES);
}
void ConvolutionUtils::conv2dBP(nd4j::LaunchContext & block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {
    BUILD_DOUBLE_SELECTOR(input->dataType(), gradO->dataType(), conv2dBP_, (block, input, weights, bias, gradO, gradI, gradW, gradB, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW), LIBND4J_TYPES, FLOAT_TYPES);
}
void ConvolutionUtils::depthwiseConv2d(nd4j::LaunchContext & block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {
    BUILD_DOUBLE_SELECTOR(input->dataType(), output->dataType(), depthwiseConv2d_, (block, input, weights, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW), LIBND4J_TYPES, FLOAT_TYPES);
}
void ConvolutionUtils::depthwiseConv2dBP(nd4j::LaunchContext & block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {
    BUILD_DOUBLE_SELECTOR(input->dataType(), gradO->dataType(), depthwiseConv2dBP_, (block, input, weights, bias, gradO, gradI, gradW, gradB, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW), LIBND4J_TYPES, FLOAT_TYPES);
}
void ConvolutionUtils::sconv2d(nd4j::LaunchContext & block, const NDArray* input, const NDArray* weightsDepth, const NDArray* weightsPoint, const NDArray* bias,  NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {
    BUILD_DOUBLE_SELECTOR(input->dataType(), output->dataType(), sconv2d_, (block, input, weightsDepth, weightsPoint, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW), LIBND4J_TYPES, FLOAT_TYPES);
}
void ConvolutionUtils::vol2col(nd4j::LaunchContext & block, const NDArray& volume, NDArray& columns, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {
    BUILD_SINGLE_SELECTOR(volume.dataType(), vol2col_, (volume, columns, sD, sH, sW, pD, pH, pW, dD, dH, dW), LIBND4J_TYPES);
}
void ConvolutionUtils::col2vol(nd4j::LaunchContext & block, const NDArray& columns, NDArray& volume, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {
    BUILD_SINGLE_SELECTOR(volume.dataType(), col2vol_, (columns, volume, sD, sH, sW, pD, pH, pW, dD, dH, dW), LIBND4J_TYPES);
}
void ConvolutionUtils::upsampling2d(nd4j::LaunchContext & block, const NDArray& input, NDArray& output, const int factorH, const int factorW, const bool isNCHW) {
    BUILD_SINGLE_SELECTOR(input.dataType(), upsampling2d_, (input, output, factorH, factorW, isNCHW), LIBND4J_TYPES);
}
void ConvolutionUtils::upsampling3d(nd4j::LaunchContext & block, const NDArray& input, NDArray& output, const int factorD, const int factorH, const int factorW, const bool isNCDHW) {
    BUILD_SINGLE_SELECTOR(input.dataType(), upsampling3d_, (input, output, factorD, factorH, factorW, isNCDHW), LIBND4J_TYPES);
}
void ConvolutionUtils::upsampling2dBP(nd4j::LaunchContext & block, const NDArray& gradO, NDArray& gradI, const bool isNCHW) {
    BUILD_SINGLE_SELECTOR(gradO.dataType(), upsampling2dBP_, (gradO, gradI, isNCHW), LIBND4J_TYPES);
}
void ConvolutionUtils::upsampling3dBP(nd4j::LaunchContext & block, const NDArray& gradO, NDArray& gradI, const bool isNCHW) {
    BUILD_SINGLE_SELECTOR(gradO.dataType(), upsampling3dBP_, (gradO, gradI, isNCHW), LIBND4J_TYPES);
}



void ConvolutionUtils::pooling2d(nd4j::LaunchContext & block, const NDArray& input, NDArray& output, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const PoolingType poolingMode, const int extraParam0) {
    BUILD_SINGLE_SELECTOR(input.dataType(), pooling2d_, (block, input, output, kH, kW, sH, sW, pH, pW, dH, dW, poolingMode, extraParam0), LIBND4J_TYPES);
}
void ConvolutionUtils::pooling3d(nd4j::LaunchContext & block, const NDArray& input, NDArray& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {
    BUILD_SINGLE_SELECTOR(input.dataType(), pooling3d_, (block, input, output, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode, extraParam0), LIBND4J_TYPES);
}
void ConvolutionUtils::pooling2dBP(nd4j::LaunchContext & block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0) {
    BUILD_SINGLE_SELECTOR(input.dataType(), pooling2dBP_, (block, input, gradO, gradI, kH, kW, sH, sW, pH, pW, dH, dW, poolingMode, extraParam0), LIBND4J_TYPES);
}
void ConvolutionUtils::pooling3dBP(nd4j::LaunchContext & block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {
    BUILD_SINGLE_SELECTOR(input.dataType(), pooling3dBP_, (block, input, gradO, gradI, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode, extraParam0), LIBND4J_TYPES);
}


BUILD_DOUBLE_TEMPLATE(template void conv2d_,            (nd4j::LaunchContext & block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW), LIBND4J_TYPES, FLOAT_TYPES);
BUILD_DOUBLE_TEMPLATE(template void conv2dBP_,          (nd4j::LaunchContext & block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW), LIBND4J_TYPES, FLOAT_TYPES);
BUILD_DOUBLE_TEMPLATE(template void depthwiseConv2d_,   (nd4j::LaunchContext & block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW), LIBND4J_TYPES, FLOAT_TYPES);
BUILD_DOUBLE_TEMPLATE(template void depthwiseConv2dBP_, (nd4j::LaunchContext & block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW), LIBND4J_TYPES, FLOAT_TYPES);
BUILD_DOUBLE_TEMPLATE(template void sconv2d_,           (nd4j::LaunchContext & block, const NDArray* input, const NDArray* weightsDepth, const NDArray* weightsPoint, const NDArray* bias,  NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW), LIBND4J_TYPES, FLOAT_TYPES);

BUILD_SINGLE_TEMPLATE(template void upsampling2d_,   (const NDArray& input, NDArray& output, const int factorH, const int factorW, const bool isNCHW), LIBND4J_TYPES);
BUILD_SINGLE_TEMPLATE(template void upsampling3d_,   (const NDArray& input, NDArray& output, const int factorD, const int factorH, const int factorW, const bool isNCDHW), LIBND4J_TYPES);
BUILD_SINGLE_TEMPLATE(template void upsampling2dBP_, (const NDArray& gradO, NDArray& gradI, const bool isNCHW), LIBND4J_TYPES);
BUILD_SINGLE_TEMPLATE(template void upsampling3dBP_, (const NDArray& gradO, NDArray& gradI, const bool isNCHW), LIBND4J_TYPES);
BUILD_SINGLE_TEMPLATE(template void vol2col_,        (const NDArray& volume, NDArray& columns, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW), LIBND4J_TYPES);
BUILD_SINGLE_TEMPLATE(template void col2vol_,        (const NDArray& columns, NDArray& volume, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW), LIBND4J_TYPES);
BUILD_SINGLE_TEMPLATE(template void pooling2d_,      (nd4j::LaunchContext & block, const NDArray& input, NDArray& output, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0), LIBND4J_TYPES);
BUILD_SINGLE_TEMPLATE(template void pooling3d_,      (nd4j::LaunchContext & block, const NDArray& input, NDArray& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0), LIBND4J_TYPES);
BUILD_SINGLE_TEMPLATE(template void pooling2dBP_,    (nd4j::LaunchContext & block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0), LIBND4J_TYPES);
BUILD_SINGLE_TEMPLATE(template void pooling3dBP_,    (nd4j::LaunchContext & block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0), LIBND4J_TYPES);

}
}
