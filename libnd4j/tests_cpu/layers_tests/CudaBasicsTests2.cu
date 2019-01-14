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
 // @author raver119@gmail.com
 //

#include "testlayers.h"
#include <NDArray.h>
#include <NDArrayFactory.h>
#include <Context.h>
#include <Node.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <specials_cuda.h>
#include <TAD.h>
#include <MmulHelper.h>

#include <cuda.h>
#include <cuda_launch_config.h>

using namespace nd4j;
using namespace nd4j::graph;

class CudaBasicsTests2 : public testing::Test {
public:

};

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_1) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::DOUBLE);
	NDArray b('f', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, nd4j::DataType::DOUBLE);
	NDArray c('f', {M,N}, nd4j::DataType::DOUBLE);

	NDArray exp('f', {M,N}, {0.1, 0.3, 0.5, 2.5, 2.7, 2.9, 4.9, 5.1, 5.3, 7.3, 7.5, 7.7, 9.7, 9.9, 10.1}, nd4j::DataType::DOUBLE);

	nd4j::MmulHelper::mmulMxM<double,double,double>(&a, &b, &c, 1., 0.);	
	// c.printIndexedBuffer();

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_2) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::DOUBLE);
	NDArray b('f', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, nd4j::DataType::DOUBLE);
	NDArray c('f', {M,N}, nd4j::DataType::DOUBLE);

	NDArray exp('f', {M,N}, {-1.6, -0.7, 0.2, -0.8, 0.1, 1., -0., 0.9, 1.8, 0.8, 1.7, 2.6, 1.6, 2.5, 3.4}, nd4j::DataType::DOUBLE);

	nd4j::MmulHelper::mmulMxM<double,double,double>(&a, &b, &c, 1., 0.);		

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_3) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::DOUBLE);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, nd4j::DataType::DOUBLE);
	NDArray c('f', {M,N}, nd4j::DataType::DOUBLE);

	NDArray exp('f', {M,N}, {-1.9, -0.9, 0.1, 1.3, 0.3, -0.7, -0.7, 0.3, 1.3, 0.1, -0.9, -1.9, 0.5, 1.5, 2.5}, nd4j::DataType::DOUBLE);

	nd4j::MmulHelper::mmulMxM<double,double,double>(&a, &b, &c, 1., 0.);	

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_4) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::DOUBLE);
	NDArray b('f', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, nd4j::DataType::DOUBLE);
	NDArray c('c', {M,N}, nd4j::DataType::DOUBLE);

	NDArray exp('c', {M,N}, {0.1, 2.5, 4.9, 7.3, 9.7,0.3, 2.7, 5.1, 7.5, 9.9,0.5, 2.9, 5.3, 7.7, 10.1}, nd4j::DataType::DOUBLE);

	nd4j::MmulHelper::mmulMxM<double,double,double>(&a, &b, &c, 1., 0.);	

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_5) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::DOUBLE);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, nd4j::DataType::DOUBLE);
	NDArray c('f', {M,N}, nd4j::DataType::DOUBLE);

	NDArray exp('f', {M,N}, {-8.8, -4.3, 0.2, 8.6, 4.1, -0.4, -8.4, -3.9, 0.6, 8.2, 3.7, -0.8, -8.0, -3.5, 1.}, nd4j::DataType::DOUBLE);

	nd4j::MmulHelper::mmulMxM<double,double,double>(&a, &b, &c, 1., 0.);	

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_6) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::DOUBLE);
	NDArray b('f', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, nd4j::DataType::DOUBLE);
	NDArray c('c', {M,N}, nd4j::DataType::DOUBLE);

	NDArray exp('c', {M,N}, {-1.6, -0.8, -0.0, 0.8, 1.6, -0.7, 0.1, 0.9, 1.7, 2.5, 0.2, 1.0, 1.8, 2.6, 3.4}, nd4j::DataType::DOUBLE);

	nd4j::MmulHelper::mmulMxM<double,double,double>(&a, &b, &c, 1., 0.);	

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_7) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::DOUBLE);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, nd4j::DataType::DOUBLE);
	NDArray c('c', {M,N}, nd4j::DataType::DOUBLE);

	NDArray exp('c', {M,N}, {-1.9, 1.3, -0.7, 0.1, 0.5, -0.9, 0.3, 0.3, -0.9, 1.5, 0.1, -0.7, 1.3, -1.9, 2.5}, nd4j::DataType::DOUBLE);

	nd4j::MmulHelper::mmulMxM<double,double,double>(&a, &b, &c, 1., 0.);	

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_8) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::DOUBLE);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, nd4j::DataType::DOUBLE);
	NDArray c('c', {M,N}, nd4j::DataType::DOUBLE);

	NDArray exp('c', {M,N}, {-8.8, 8.6, -8.4, 8.2, -8.0, -4.3, 4.1, -3.9, 3.7, -3.5, 0.2, -0.4, 0.6, -0.8, 1.}, nd4j::DataType::DOUBLE);

	nd4j::MmulHelper::mmulMxM<double,double,double>(&a, &b, &c, 1., 0.);	

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_9) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::FLOAT32);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, nd4j::DataType::FLOAT32);
	NDArray c('c', {M,N}, nd4j::DataType::FLOAT32);

	NDArray exp('c', {M,N}, {-8.8, 8.6, -8.4, 8.2, -8.0, -4.3, 4.1, -3.9, 3.7, -3.5, 0.2, -0.4, 0.6, -0.8, 1.}, nd4j::DataType::FLOAT32);

	nd4j::MmulHelper::mmulMxM<float,float,float>(&a, &b, &c, 1., 0.);	

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_10) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::FLOAT32);
	NDArray b('f', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, nd4j::DataType::FLOAT32);
	NDArray c('f', {M,N}, nd4j::DataType::FLOAT32);

	NDArray exp('f', {M,N}, {0.1, 0.3, 0.5, 2.5, 2.7, 2.9, 4.9, 5.1, 5.3, 7.3, 7.5, 7.7, 9.7, 9.9, 10.1}, nd4j::DataType::FLOAT32);

	nd4j::MmulHelper::mmulMxM<float,float,float>(&a, &b, &c, 1., 0.);	
	// c.printIndexedBuffer();

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_11) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::FLOAT32);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, nd4j::DataType::FLOAT32);
	NDArray c('f', {M,N}, nd4j::DataType::FLOAT32);

	NDArray exp('f', {M,N}, {-1.9, -0.9, 0.1, 1.3, 0.3, -0.7, -0.7, 0.3, 1.3, 0.1, -0.9, -1.9, 0.5, 1.5, 2.5}, nd4j::DataType::FLOAT32);

	nd4j::MmulHelper::mmulMxM<float,float,float>(&a, &b, &c, 1., 0.);	

	ASSERT_TRUE(c.equalsTo(&exp));
}

// //////////////////////////////////////////////////////////////////////////
// TEST_F(CudaBasicsTests2, mmulMxM_12) {

// 	const Nd4jLong M = 3;
// 	const Nd4jLong K = 4;
// 	const Nd4jLong N = 5;

// 	NDArray a('c', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::HALF);
// 	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, nd4j::DataType::HALF);
// 	NDArray c('c', {M,N}, nd4j::DataType::HALF);

// 	NDArray exp('c', {M,N}, {-8.8, 8.6, -8.4, 8.2, -8.0, -4.3, 4.1, -3.9, 3.7, -3.5, 0.2, -0.4, 0.6, -0.8, 1.}, nd4j::DataType::HALF);

// 	nd4j::MmulHelper::mmulMxM<float16,float16,float16>(&a, &b, &c, 1., 0.);	

// 	ASSERT_TRUE(c.equalsTo(&exp));
// }

// //////////////////////////////////////////////////////////////////////////
// TEST_F(CudaBasicsTests2, mmulMxM_13) {

// 	const Nd4jLong M = 3;
// 	const Nd4jLong K = 4;
// 	const Nd4jLong N = 5;

// 	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::HALF);
// 	NDArray b('f', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, nd4j::DataType::HALF);
// 	NDArray c('f', {M,N}, nd4j::DataType::HALF);

// 	NDArray exp('f', {M,N}, {0.1, 0.3, 0.5, 2.5, 2.7, 2.9, 4.9, 5.1, 5.3, 7.3, 7.5, 7.7, 9.7, 9.9, 10.1}, nd4j::DataType::HALF);

// 	nd4j::MmulHelper::mmulMxM<float16,float16,float16>(&a, &b, &c, 1., 0.);	
// 	// c.printIndexedBuffer();

// 	ASSERT_TRUE(c.equalsTo(&exp));
// }

// //////////////////////////////////////////////////////////////////////////
// TEST_F(CudaBasicsTests2, mmulMxM_14) {

// 	const Nd4jLong M = 3;
// 	const Nd4jLong K = 4;
// 	const Nd4jLong N = 5;

// 	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::HALF);
// 	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, nd4j::DataType::HALF);
// 	NDArray c('f', {M,N}, nd4j::DataType::HALF);

// 	NDArray exp('f', {M,N}, {-1.9, -0.9, 0.1, 1.3, 0.3, -0.7, -0.7, 0.3, 1.3, 0.1, -0.9, -1.9, 0.5, 1.5, 2.5}, nd4j::DataType::HALF);

// 	nd4j::MmulHelper::mmulMxM<float16,float16,float16>(&a, &b, &c, 1., 0.);	

// 	ASSERT_TRUE(c.equalsTo(&exp));
// }