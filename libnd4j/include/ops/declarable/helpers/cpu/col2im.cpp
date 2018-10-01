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
// Created by raver119 on 30.11.17.
//

#include <ops/declarable/helpers/col2im.h>

namespace nd4j {
namespace ops {
namespace helpers {

// [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]
template <typename T>
void col2im_(graph::LaunchContext& context, const NDArray& input,  NDArray& output, const int sH, const int sW, const int pH, const int pW, const int iH, const int iW, const int dH, const int dW) {

    auto imBuff         = output.bufferAsT<T>();
	auto colBuff        = input.bufferAsT<T>();
	auto imShapeBuffer  = output.getShapeInfo();
	auto colShapeBuffer = input.getShapeInfo();
    auto colShape  		= shape::shapeOf(colShapeBuffer);
    auto colStride 		= shape::stride(colShapeBuffer);
    auto imShape  	    = shape::shapeOf(imShapeBuffer);
    auto imStride 	    = shape::stride(imShapeBuffer);            

    const int bS = imShape[0];
    const int iC = imShape[1];
    const int kH = colShape[2];
    const int kW = colShape[3];                    
    const int oH = colShape[4];
    const int oW = colShape[5];
    const Nd4jLong colStride0 = colStride[0];
    const Nd4jLong colStride1 = colStride[1];
    const Nd4jLong colStride2 = colStride[2];
    const Nd4jLong colStride3 = colStride[3];
    const Nd4jLong colStride4 = colStride[4];
    const Nd4jLong colStride5 = colStride[5];
    const Nd4jLong imStride0  = imStride[0];
    const Nd4jLong imStride1  = imStride[1];
    const Nd4jLong imStride2  = imStride[2];
    const Nd4jLong imStride3  = imStride[3];

    // initial zeroing of image content
    const auto imEWS = shape::elementWiseStride(imShapeBuffer);
    if(imEWS == 1) {
        memset(imBuff, 0, shape::length(imShapeBuffer) * sizeof(T));
    } else if (imEWS > 1) {
#pragma omp parallel for schedule(static) proc_bind(close)
        for (int i = 0; i < shape::length(imShapeBuffer) * imEWS; i += imEWS)
            imBuff[i] = static_cast<T>(0.f);
    } else {
        Nd4jLong idx[MAX_RANK];
        const auto len = shape::length(imShapeBuffer);
#pragma omp parallel for schedule(static) proc_bind(close) private(idx)
        for (int i = 0; i < len; i++) {
            shape::ind2subC(shape::rank(imShapeBuffer), imShape, i, len, idx);
            auto imOffset = shape::getOffset(0, imShape, imStride, idx, shape::rank(imShapeBuffer));

            imBuff[imOffset] = 0;
        }
    }
            
	T *col, *im;
    int imRow, imCol;

    if (shape::order(colShapeBuffer) == 'c' &&  shape::order(imShapeBuffer) == 'c' && shape::strideDescendingCAscendingF(colShapeBuffer) && shape::strideDescendingCAscendingF(imShapeBuffer)) {
            
#pragma omp parallel for schedule(static) proc_bind(close) private(col, im, imRow, imCol)
    	for (int b = 0; b < bS; b++) {        
      		for (int c = 0; c < iC; ++c) {                    
            	for (int kRow = 0; kRow < kH; ++kRow) {                        
                	for (int kCol = 0; kCol < kW; ++kCol) {                            
                    	for (int colH = 0; colH < oH; ++colH) {
                        	for (int colW = 0; colW < oW; ++colW) {                    

                            	imRow = (-pH + kRow * dH) + colH*sH;
                                imCol = (-pW + kCol * dW) + colW*sW;

                                col = colBuff + b*colStride0 + c*colStride1 + kRow*colStride2 + kCol*colStride3 + colH*colStride4 + colW*colStride5;
                                im  = imBuff  + b*imStride0  + c*imStride1  + imRow*imStride2 + imCol*imStride3;

                                if (static_cast<unsigned>(imRow) < static_cast<unsigned>(iH) && static_cast<unsigned>(imCol) < static_cast<unsigned>(iW))
                                	*im += *col;
                            }
                        }
                    }
                }
            }
        }  
    }
    else {

#pragma omp parallel for schedule(static) proc_bind(close) private(im, col, imRow, imCol)
    	for (int b = 0; b < bS; b++) {        
        	for (int colH = 0; colH < oH; ++colH) {
            	for (int colW = 0; colW < oW; ++colW) {
                	for (int c = 0; c < iC; ++c) {                        
                    	for (int kRow = 0; kRow < kH; ++kRow) {                        
                        	for (int kCol = 0; kCol < kW; ++kCol) {                            
                        
                            	imRow = (-pH + kRow * dH) + colH*sH;
                                imCol = (-pW + kCol * dW) + colW*sW;
                                        
                                col = colBuff + b*colStride0 + c*colStride1 + kRow*colStride2 + kCol*colStride3 + colH*colStride4 + colW*colStride5;
                                im  = imBuff  + b*imStride0  + c*imStride1  + imRow*imStride2 + imCol*imStride3;

                                if (static_cast<unsigned>(imRow) < static_cast<unsigned>(iH) && static_cast<unsigned>(imCol) < static_cast<unsigned>(iW))
                                	*im += *col;
                            }
                        }
                    }
                }                           
            }
        }  
    }
}


void col2im(graph::LaunchContext& context, const NDArray& input,  NDArray& output, const int sH, const int sW, const int pH, const int pW, const int iH, const int iW, const int dH, const int dW) {
	BUILD_SINGLE_SELECTOR(input.dataType(), col2im_, (context, input, output, sH, sW, pH, pW, iH, iW, dH, dW), LIBND4J_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void col2im_, (nd4j::graph::LaunchContext& context, const NDArray& input,  NDArray& output, const int sH, const int sW, const int pH, const int pW, const int iH, const int iW, const int dH, const int dW), LIBND4J_TYPES);

}
}
}
