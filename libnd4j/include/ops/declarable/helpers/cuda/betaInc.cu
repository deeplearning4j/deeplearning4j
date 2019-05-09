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
// Created by Yurii Shyrma on 11.12.2017
//

#include<cmath> 
#include <DataTypeUtils.h>
#include<ops/declarable/helpers/betaInc.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

const int maxIter = 10000;				// max number of loop iterations in function for continued fractions 
const int maxValue = 3000;				// if a and b are both > maxValue, then apply Gauss-Legendre quadrature.


// 18 values of abscissas and weights for 36-point Gauss-Legendre integration,
// take a note - weights and abscissas are symmetric around the midpoint of the range of integration: 36/2 = 18
const double abscissas[18] = {0.0021695375159141994,
0.011413521097787704,0.027972308950302116,0.051727015600492421,
0.082502225484340941, 0.12007019910960293,0.16415283300752470,
0.21442376986779355, 0.27051082840644336, 0.33199876341447887,
0.39843234186401943, 0.46931971407375483, 0.54413605556657973,
0.62232745288031077, 0.70331500465597174, 0.78649910768313447,
0.87126389619061517, 0.95698180152629142};
const double weights[18] = {0.0055657196642445571,
0.012915947284065419,0.020181515297735382,0.027298621498568734,
0.034213810770299537,0.040875750923643261,0.047235083490265582,
0.053244713977759692,0.058860144245324798,0.064039797355015485,
0.068745323835736408,0.072941885005653087,0.076598410645870640,
0.079687828912071670,0.082187266704339706,0.084078218979661945,
0.085346685739338721,0.085983275670394821};




///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
// overload betaInc for arrays, shapes of a, b and x must be the same !!!
NDArray betaInc(nd4j::LaunchContext * context, const NDArray& a, const NDArray& b, const NDArray& x) {
	auto xType = a.dataType();
	//BUILD_SINGLE_SELECTOR(xType, return betaIncT, (a, b, x), FLOAT_TYPES);
	return a;
}


}
}
}

