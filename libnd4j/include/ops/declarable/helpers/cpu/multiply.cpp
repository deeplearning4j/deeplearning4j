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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 30.07.2018
//

#include <ops/declarable/helpers/multiply.h>
#include <ShapeUtils.h>

namespace nd4j {
namespace ops {
namespace helpers {


template <typename T>
void multiplyBP(const NDArray<T>& x, const NDArray<T>& y, const NDArray<T>& dLdz, NDArray<T>& dLdx, NDArray<T>& dLdy) {

	const Nd4jLong xLen = x.lengthOf();
	const Nd4jLong yLen = y.lengthOf();
    
    if(xLen == 1 && yLen == 1) {	// both are scalars
    	dLdx(0.) = y(0.) * dLdz(0.);
    	dLdy(0.) = x(0.) * dLdz(0.);
    }
    else if(xLen == 1) { 			// x is scalar and y is not 

  		dLdx(0.) = (y * dLdz).template reduceNumber<simdOps::Sum<T>>();    	
  		dLdy.assign(dLdz * x(0.));		
    }
    else if(yLen == 1) { 			// y is scalar and x is not 

    	dLdy(0.) = (x * dLdz).template reduceNumber<simdOps::Sum<T>>();
    	dLdx.assign(dLdz * y(0.));
    }    
    else if(x.isSameShape(&y)) {

    	dLdx.assign(y * dLdz);
    	dLdy.assign(x * dLdz);
    }
    else if (x.isSameShape(&dLdz)) {
    	
    	const Nd4jLong zLen = dLdz.lengthOf();
    	const Nd4jLong* yShapeInfo = y.getShapeInfo();
    	const Nd4jLong* zShapeInfo = dLdz.getShapeInfo();
    	
    	dLdy = (T)0;

// #pragma omp parallel for if(zLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
#pragma omp parallel for schedule(guided)
    	for(Nd4jLong i = 0; i < zLen; ++i) {
            
            const T dLdzVal = dLdz(i);
        	const Nd4jLong yInd = ShapeUtils<T>::getSubArrayIndex(zShapeInfo, yShapeInfo, i);        	
        	dLdx(i) =  y(yInd) * dLdzVal;
#pragma omp critical          		
        		dLdy(yInd) += x(i) * dLdzVal;
    	}
    } 
    else if (y.isSameShape(&dLdz)) {

    	const Nd4jLong zLen = dLdz.lengthOf();
    	const Nd4jLong* xShapeInfo = x.getShapeInfo();
    	const Nd4jLong* zShapeInfo = dLdz.getShapeInfo();
    	
    	dLdx = (T)0;

// #pragma omp parallel for if(zLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
#pragma omp parallel for schedule(guided)
    	for(Nd4jLong i = 0; i < zLen; ++i) {
            
            const T dLdzVal = dLdz(i);
        	const Nd4jLong xInd = ShapeUtils<T>::getSubArrayIndex(zShapeInfo, xShapeInfo, i);        	
        	dLdy(i) =  x(xInd) * dLdzVal;
#pragma omp critical          		
        		dLdx(xInd) += y(i) * dLdzVal;
    	}
    }
    else {

    	const Nd4jLong zLen = dLdz.lengthOf();
    	const Nd4jLong* xShapeInfo = x.getShapeInfo();
    	const Nd4jLong* yShapeInfo = y.getShapeInfo();
    	const Nd4jLong* zShapeInfo = dLdz.getShapeInfo();

    	dLdx = (T)0;
    	dLdy = (T)0;

// #pragma omp parallel for if(zLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
#pragma omp parallel for schedule(guided)
    	for(Nd4jLong i = 0; i < zLen; ++i) {
            
            const T dLdzVal = dLdz(i);
        	const Nd4jLong xInd = ShapeUtils<T>::getSubArrayIndex(zShapeInfo, xShapeInfo, i);
        	const Nd4jLong yInd = ShapeUtils<T>::getSubArrayIndex(zShapeInfo, yShapeInfo, i);        	
#pragma omp critical  
			{        	
        		dLdx(xInd) += y(yInd) * dLdzVal;
        		dLdy(yInd) += x(xInd) * dLdzVal;
        	}
    	}
    }
}

template void multiplyBP(const NDArray<float16>& x, const NDArray<float16>& y, const NDArray<float16>& dLdz, NDArray<float16>& dLdx, NDArray<float16>& dLdy);
template void multiplyBP(const NDArray<float>& x, const NDArray<float>& y, const NDArray<float>& dLdz, NDArray<float>& dLdx, NDArray<float>& dLdy);
template void multiplyBP(const NDArray<double>& x, const NDArray<double>& y, const NDArray<double>& dLdz, NDArray<double>& dLdx, NDArray<double>& dLdy);


}
}
}
