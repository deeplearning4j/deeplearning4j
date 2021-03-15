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
 // @author raver119@gmail.com
 //

#include "testlayers.h"
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <graph/Context.h>
#include <graph/Node.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <ops/specials_cuda.h>
#include <helpers/TAD.h>
#include <helpers/MmulHelper.h>

#include <cuda.h>

using namespace sd;
using namespace sd::graph;

class CudaBasicsTests2 : public testing::Test {
public:

};

TEST_F(CudaBasicsTests2, test_devices_1) {
	auto caps = Environment::getInstance().capabilities();
	ASSERT_FALSE(caps.empty());
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_1) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::FLOAT32);
	NDArray b('f', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::FLOAT32);
	NDArray c('f', {M,N}, sd::DataType::FLOAT32);

	NDArray exp('f', {M,N}, {0.1, 0.3, 0.5, 2.5, 2.7, 2.9, 4.9, 5.1, 5.3, 7.3, 7.5, 7.7, 9.7, 9.9, 10.1}, sd::DataType::FLOAT32);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);
	// c.printIndexedBuffer();

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_2) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
	NDArray b('f', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::DOUBLE);
	NDArray c('f', {M,N}, sd::DataType::DOUBLE);
	NDArray exp('f', {M,N}, {-1.6, -0.7, 0.2, -0.8, 0.1, 1., -0., 0.9, 1.8, 0.8, 1.7, 2.6, 1.6, 2.5, 3.4}, sd::DataType::DOUBLE);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_3) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::DOUBLE);
	NDArray c('f', {M,N}, sd::DataType::DOUBLE);

	NDArray exp('f', {M,N}, {-1.9, -0.9, 0.1, 1.3, 0.3, -0.7, -0.7, 0.3, 1.3, 0.1, -0.9, -1.9, 0.5, 1.5, 2.5}, sd::DataType::DOUBLE);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_4) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
	NDArray b('f', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::DOUBLE);
	NDArray c('c', {M,N}, sd::DataType::DOUBLE);

	NDArray exp('c', {M,N}, {0.1, 2.5, 4.9, 7.3, 9.7,0.3, 2.7, 5.1, 7.5, 9.9,0.5, 2.9, 5.3, 7.7, 10.1}, sd::DataType::DOUBLE);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);
	ASSERT_TRUE(c.equalsTo(&exp));


	// NDArray* pA = a.permute({1,0});
	// NDArray* pB = b.permute({1,0});
	// NDArray* pC = c.permute({1,0});

	// sd::MmulHelper::mmul(pB, pA, pC, 1., 0.);
	// ASSERT_TRUE(c.equalsTo(&exp));

	// delete pA;
	// delete pB;
	// delete pC;
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_5) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::DOUBLE);
	NDArray c('f', {M,N}, sd::DataType::DOUBLE);

	NDArray exp('f', {M,N}, {-8.8, -4.3, 0.2, 8.6, 4.1, -0.4, -8.4, -3.9, 0.6, 8.2, 3.7, -0.8, -8.0, -3.5, 1.}, sd::DataType::DOUBLE);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_6) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
	NDArray b('f', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::DOUBLE);
	NDArray c('c', {M,N}, sd::DataType::DOUBLE);

	NDArray exp('c', {M,N}, {-1.6, -0.8, -0.0, 0.8, 1.6, -0.7, 0.1, 0.9, 1.7, 2.5, 0.2, 1.0, 1.8, 2.6, 3.4}, sd::DataType::DOUBLE);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_7) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::DOUBLE);
	NDArray c('c', {M,N}, sd::DataType::DOUBLE);

	NDArray exp('c', {M,N}, {-1.9, 1.3, -0.7, 0.1, 0.5, -0.9, 0.3, 0.3, -0.9, 1.5, 0.1, -0.7, 1.3, -1.9, 2.5}, sd::DataType::DOUBLE);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_8) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::DOUBLE);
	NDArray c('c', {M,N}, sd::DataType::DOUBLE);

	NDArray exp('c', {M,N}, {-8.8, 8.6, -8.4, 8.2, -8.0, -4.3, 4.1, -3.9, 3.7, -3.5, 0.2, -0.4, 0.6, -0.8, 1.}, sd::DataType::DOUBLE);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_9) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::FLOAT32);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::FLOAT32);
	NDArray c('c', {M,N}, sd::DataType::FLOAT32);

	NDArray exp('c', {M,N}, {-8.8, 8.6, -8.4, 8.2, -8.0, -4.3, 4.1, -3.9, 3.7, -3.5, 0.2, -0.4, 0.6, -0.8, 1.}, sd::DataType::FLOAT32);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_10) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::FLOAT32);
	NDArray b('f', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::FLOAT32);
	NDArray c('f', {M,N}, sd::DataType::FLOAT32);

	NDArray exp('f', {M,N}, {0.1, 0.3, 0.5, 2.5, 2.7, 2.9, 4.9, 5.1, 5.3, 7.3, 7.5, 7.7, 9.7, 9.9, 10.1}, sd::DataType::FLOAT32);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);
	// c.printIndexedBuffer();

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_11) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::FLOAT32);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::FLOAT32);
	NDArray c('f', {M,N}, sd::DataType::FLOAT32);

	NDArray exp('f', {M,N}, {-1.9, -0.9, 0.1, 1.3, 0.3, -0.7, -0.7, 0.3, 1.3, 0.1, -0.9, -1.9, 0.5, 1.5, 2.5}, sd::DataType::FLOAT32);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_12) {

	int devCnt = 0;
	cudaGetDevice(&devCnt);
	if(Environment::getInstance().capabilities()[devCnt].first() < 5) return;

	const Nd4jLong M = 4;
	const Nd4jLong K = 4;
	const Nd4jLong N = 4;

	NDArray a('f', {M,K}, {1.,2,3,4,5,6,7,8,9,2,3,2,1,0,4,7.}, sd::DataType::INT8);
	NDArray b('f', {K,N}, {-2,-3,0,1,5,-6,7,-8,9,-1,2,-2,3,-4,5,-6.}, sd::DataType::INT8);
	NDArray c('f', {M,N}, sd::DataType::FLOAT32);

	NDArray exp('f', {M,N}, {-16., -22., -23., -25., 30., -12., -38., -70., 20., 16., 18., 18., 22., -8., -28., -52.}, sd::DataType::FLOAT32);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);
	// c.printBuffer();

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_13) {

	int devCnt = 0;
	cudaGetDevice(&devCnt);
	if(Environment::getInstance().capabilities()[devCnt].first() < 5) return;

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('f', {M,K}, {1.,2,3,4,5,6,7,8,9,10,11,12}, sd::DataType::INT8);
	NDArray b('c', {K,N}, {-2,-3,0,1,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::INT8);
	NDArray c('f', {M,N}, sd::DataType::FLOAT32);

	NDArray exp('f', {M,N}, {-109., -122., -135., 111., 120., 129., -121., -134., -147., 129., 144., 159., -130., -140., -150.}, sd::DataType::FLOAT32);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_14) {

	int devCnt = 0;
	cudaGetDevice(&devCnt);
	if(Environment::getInstance().capabilities()[devCnt].first() < 5) return;

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.,2,3,4,5,6,7,8,9,10,11,12}, sd::DataType::INT8);
	NDArray b('c', {K,N}, {-2,-3,0,1,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::INT8);
	NDArray c('c', {M,N}, sd::DataType::FLOAT32);

	NDArray exp('c', {M,N}, {-45., 43., -49., 53., -50., -97., 79., -101., 113., -90., -149., 115., -153., 173., -130.}, sd::DataType::FLOAT32);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_15) {

	int devCnt = 0;
	cudaGetDevice(&devCnt);
	if(Environment::getInstance().capabilities()[devCnt].first() < 5) return;

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::HALF);
	NDArray b('f', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::HALF);
	NDArray c('f', {M,N}, sd::DataType::FLOAT32);

	NDArray exp('f', {M,N}, {0.1, 0.3, 0.5, 2.5, 2.7, 2.9, 4.9, 5.1, 5.3, 7.3, 7.5, 7.7, 9.7, 9.9, 10.1}, sd::DataType::FLOAT32);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);
	// c.printBuffer();

	ASSERT_TRUE(c.equalsTo(&exp, 0.01));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_16) {

	int devCnt = 0;
	cudaGetDevice(&devCnt);
	if(Environment::getInstance().capabilities()[devCnt].first() < 5) return;

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::HALF);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::HALF);
	NDArray c('f', {M,N}, sd::DataType::FLOAT32);

	NDArray exp('f', {M,N}, {-1.9, -0.9, 0.1, 1.3, 0.3, -0.7, -0.7, 0.3, 1.3, 0.1, -0.9, -1.9, 0.5, 1.5, 2.5}, sd::DataType::FLOAT32);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp, 0.01));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_17) {

	int devCnt = 0;
	cudaGetDevice(&devCnt);
	if(Environment::getInstance().capabilities()[devCnt].first() < 5) return;

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::HALF);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::HALF);
	NDArray c('c', {M,N}, sd::DataType::FLOAT32);

	NDArray exp('c', {M,N}, {-8.8, 8.6, -8.4, 8.2, -8.0, -4.3, 4.1, -3.9, 3.7, -3.5, 0.2, -0.4, 0.6, -0.8, 1.}, sd::DataType::FLOAT32);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp, 0.01));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_18) {

	int devCnt = 0;
	cudaGetDevice(&devCnt);
	if(Environment::getInstance().capabilities()[devCnt].first() < 5.3) return;

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::HALF);
	NDArray b('f', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::HALF);
	NDArray c('f', {M,N}, sd::DataType::HALF);

	NDArray exp('f', {M,N}, {0.1, 0.3, 0.5, 2.5, 2.7, 2.9, 4.9, 5.1, 5.3, 7.3, 7.5, 7.7, 9.7, 9.9, 10.1}, sd::DataType::HALF);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp, 1e-1));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_19) {

	int devCnt = 0;
	cudaGetDevice(&devCnt);
	if(Environment::getInstance().capabilities()[devCnt].first() < 5.3) return;

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::HALF);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::HALF);
	NDArray c('f', {M,N}, sd::DataType::HALF);

	NDArray exp('f', {M,N}, {-1.9, -0.9, 0.1, 1.3, 0.3, -0.7, -0.7, 0.3, 1.3, 0.1, -0.9, -1.9, 0.5, 1.5, 2.5}, sd::DataType::HALF);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp, 1e-1));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_20) {

	int devCnt = 0;
	cudaGetDevice(&devCnt);
	if(Environment::getInstance().capabilities()[devCnt].first() < 5.3) return;

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::HALF);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::HALF);
	NDArray c('c', {M,N}, sd::DataType::HALF);

	NDArray exp('c', {M,N}, {-8.8, 8.6, -8.4, 8.2, -8.0, -4.3, 4.1, -3.9, 3.7, -3.5, 0.2, -0.4, 0.6, -0.8, 1.}, sd::DataType::HALF);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp, 1e-1));
}
/*
//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_21) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.,2,3,4,5,6,7,8,9,10,11,12}, sd::DataType::INT8);
	NDArray b('c', {K,N}, {-2,-3,0,1,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::FLOAT32);
	NDArray c('c', {M,N}, sd::DataType::DOUBLE);

	NDArray exp('c', {M,N}, {-45., 43., -49., 53., -50., -97., 79., -101., 113., -90., -149., 115., -153., 173., -130.}, sd::DataType::DOUBLE);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_22) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.,2,3,4,5,6,7,8,9,10,11,12}, sd::DataType::FLOAT32);
	NDArray b('c', {K,N}, {-2,-3,0,1,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::HALF);
	NDArray c('c', {M,N}, sd::DataType::FLOAT32);

	NDArray exp('c', {M,N}, {-45., 43., -49., 53., -50., -97., 79., -101., 113., -90., -149., 115., -153., 173., -130.}, sd::DataType::FLOAT32);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);
	// c.printBuffer();

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_23) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::HALF);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::HALF);
	NDArray c('c', {M,N}, sd::DataType::DOUBLE);

	NDArray exp('c', {M,N}, {-8.8, 8.6, -8.4, 8.2, -8.0, -4.3, 4.1, -3.9, 3.7, -3.5, 0.2, -0.4, 0.6, -0.8, 1.}, sd::DataType::DOUBLE);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp, 0.01));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_24) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::HALF);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::FLOAT32);
	NDArray c('c', {M,N}, sd::DataType::DOUBLE);

	NDArray exp('c', {M,N}, {-8.8, 8.6, -8.4, 8.2, -8.0, -4.3, 4.1, -3.9, 3.7, -3.5, 0.2, -0.4, 0.6, -0.8, 1.}, sd::DataType::DOUBLE);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp, 0.01));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_25) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
	NDArray b('c', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::FLOAT32);
	NDArray c('c', {M,N}, sd::DataType::HALF);

	NDArray exp('c', {M,N}, {-8.8, 8.6, -8.4, 8.2, -8.0, -4.3, 4.1, -3.9, 3.7, -3.5, 0.2, -0.4, 0.6, -0.8, 1.}, sd::DataType::HALF);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp, 0.01));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_26) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	// 3x4 * 4x5 = 3x5
	NDArray a('c', {M,K}, {1.,2,3,4,5,6,7,8,9,10,11,12}, sd::DataType::INT64);
	NDArray b('c', {K,N}, {-2,-3,0,1,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::FLOAT32);
	NDArray c('c', {M,N}, sd::DataType::DOUBLE);

	NDArray exp('c', {M,N}, {-45., 43., -49., 53., -50., -97., 79., -101., 113., -90., -149., 115., -153., 173., -130.}, sd::DataType::DOUBLE);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);
	// c.printBuffer();

	ASSERT_TRUE(c.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_27) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('f', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::HALF);
	NDArray b('f', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::DOUBLE);
	NDArray c('f', {M,N}, sd::DataType::FLOAT32);

	NDArray exp('f', {M,N}, {0.1, 0.3, 0.5, 2.5, 2.7, 2.9, 4.9, 5.1, 5.3, 7.3, 7.5, 7.7, 9.7, 9.9, 10.1}, sd::DataType::FLOAT32);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);
	// c.printBuffer();

	ASSERT_TRUE(c.equalsTo(&exp, 0.01));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxM_28) {

	const Nd4jLong M = 3;
	const Nd4jLong K = 4;
	const Nd4jLong N = 5;

	NDArray a('c', {M,K}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::FLOAT32);
	NDArray b('f', {K,N}, {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16,17,-18,19,-20}, sd::DataType::DOUBLE);
	NDArray c('f', {M,N}, sd::DataType::FLOAT32);

	NDArray exp('f', {M,N}, {-1.6, -0.7, 0.2, -0.8, 0.1, 1., -0., 0.9, 1.8, 0.8, 1.7, 2.6, 1.6, 2.5, 3.4}, sd::DataType::FLOAT32);

	sd::MmulHelper::mmul(&a, &b, &c, 1., 0.);

	ASSERT_TRUE(c.equalsTo(&exp));
}
 */

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_1) {

	const Nd4jLong M = 3;
	const Nd4jLong N = 4;

	NDArray a('f', {M,N}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
	NDArray x('f', {N}, {1,-2,3,-4}, sd::DataType::DOUBLE);
	NDArray y('f', {M}, sd::DataType::DOUBLE);

	NDArray exp('f', {M}, {0.1, 0.3, 0.5}, sd::DataType::DOUBLE);

	sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
	ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_2) {

	const Nd4jLong M = 3;
	const Nd4jLong N = 4;

	NDArray a('c', {M,N}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
	NDArray x('f', {N}, {1,-2,3,-4}, sd::DataType::DOUBLE);
	NDArray y('f', {M}, sd::DataType::DOUBLE);

	NDArray exp('f', {M}, {-1.6, -0.7, 0.2}, sd::DataType::DOUBLE);

	sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
	ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_3) {

	const Nd4jLong M = 3;
	const Nd4jLong N = 4;

	NDArray a('c', {M,N}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
	NDArray x('c', {N}, {1,-2,3,-4}, sd::DataType::DOUBLE);
	NDArray y('f', {M}, sd::DataType::DOUBLE);

	NDArray exp('f', {M}, {-1.6, -0.7, 0.2}, sd::DataType::DOUBLE);

	sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
	ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_4) {

	const Nd4jLong M = 3;
	const Nd4jLong N = 4;

	NDArray a('c', {M,N}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
	NDArray x('c', {N}, {1,-2,3,-4}, sd::DataType::DOUBLE);
	NDArray y('c', {M}, sd::DataType::DOUBLE);

	NDArray exp('c', {M}, {-1.6, -0.7, 0.2}, sd::DataType::DOUBLE);

	sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
	ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_5) {

	const Nd4jLong M = 3;
	const Nd4jLong N = 4;

	NDArray a('f', {M,N}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
	NDArray x('c', {N}, {1,-2,3,-4}, sd::DataType::DOUBLE);
	NDArray y('c', {M}, sd::DataType::DOUBLE);

	NDArray exp('c', {M}, {0.1, 0.3, 0.5}, sd::DataType::DOUBLE);

	sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
	ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_6) {

	const Nd4jLong M = 3;
	const Nd4jLong N = 4;

	NDArray a('f', {M,N}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
	NDArray temp('f', {M,N,5}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
	NDArray x = temp(6, {0,2});
	NDArray y('f', {M}, sd::DataType::DOUBLE);

	NDArray exp('f', {M}, {5.5, 5.1, 4.7}, sd::DataType::DOUBLE);

	sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
	ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_7) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('f', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
    a.permutei({1,0});
    NDArray temp('f', {M,N,5}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray x = temp(6, {0,2});
    NDArray y('f', {M}, sd::DataType::DOUBLE);

    NDArray exp('f', {M}, {5.1, 3.3, 1.5}, sd::DataType::DOUBLE);

    sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_8) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('f', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
    a.permutei({1,0});
    NDArray temp('f', {N,M,5}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray x = temp(4, {1,2});
    NDArray y('f', {M}, sd::DataType::DOUBLE);

    NDArray exp('f', {M}, {6.2, 4.5, 1.7}, sd::DataType::DOUBLE);

    sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_9) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('f', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
    a.permutei({1,0});
    NDArray temp('f', {5,M,N}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray x = temp(3, {0,1});
    NDArray y('f', {M}, sd::DataType::DOUBLE);

    NDArray exp('f', {M}, {1.5, 1.8, 1.5}, sd::DataType::DOUBLE);

    sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_10) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('c', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
    a.permutei({1,0});
    NDArray temp('f', {5,M,N}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray x = temp(2, {0,1});
    NDArray y('f', {M}, sd::DataType::DOUBLE);

    NDArray exp('f', {M}, {-0.3, 0.3, 0.9}, sd::DataType::DOUBLE);

    sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_11) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('c', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
    a.permutei({1,0});
    NDArray temp('c', {5,N,M}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray x = temp(13, {0,2});
    NDArray y('f', {M}, sd::DataType::DOUBLE);

    NDArray exp('f', {M}, {-12.1, -10.9, -9.7}, sd::DataType::DOUBLE);

    sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_12) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('c', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
    a.permutei({1,0});
    NDArray temp('c', {5,N,M}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray x = temp(10, {0,2});
    NDArray y('c', {M}, sd::DataType::DOUBLE);

    NDArray exp('c', {M}, {3.3, 3.3, 3.3}, sd::DataType::DOUBLE);

    sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_13) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('c', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
    a.permutei({1,0});
    NDArray temp('f', {5,M,N}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray x = temp(2, {0,1}, true);
    NDArray y('f', {M}, sd::DataType::DOUBLE);

    NDArray exp('f', {M}, {-0.3, 0.3, 0.9}, sd::DataType::DOUBLE);

    sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_14) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('c', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
    a.permutei({1,0});
    NDArray temp('c', {5,N,M}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray x = temp(10, {0,2}, true);
    NDArray y('c', {M}, sd::DataType::DOUBLE);

    NDArray exp('c', {M}, {3.3, 3.3, 3.3}, sd::DataType::DOUBLE);

    sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_15) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('c', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
    a.permutei({1,0});
    NDArray temp('f', {5,M,N}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray x = temp(2, {0,1});
    NDArray y = temp(17, {0,2});

    NDArray exp('f', {M}, {-0.3, 0.3, 0.9}, sd::DataType::DOUBLE);

    sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_16) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('c', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
    a.permutei({1,0});
    NDArray temp('f',  {5,M,N}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray temp1('c', {5,M,N}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray x = temp(2, {0,1});
    NDArray y = temp1(17, {0,2});

    NDArray exp('c', {M}, {-0.3, 0.3, 0.9}, sd::DataType::DOUBLE);

    sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_17) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('c', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
    a.permutei({1,0});
    NDArray temp('f', {5,M,N}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray x = temp(2, {0,1});
    NDArray y = temp(17, {0,2}, true);
    // y.printShapeInfo();

    NDArray exp('f', {1,M,1}, {-0.3, 0.3, 0.9}, sd::DataType::DOUBLE);

    sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_18) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('c', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::DOUBLE);
    a.permutei({1,0});
    NDArray temp('f',  {5,M,N}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray temp1('c', {5,M,N}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray x = temp(2, {0,1},true);
    NDArray y = temp1(17, {0,2},true);

    NDArray exp('c', {1,M,1}, {-0.3, 0.3, 0.9}, sd::DataType::DOUBLE);

    sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
/*
TEST_F(CudaBasicsTests2, mmulMxV_19) {

	const Nd4jLong M = 3;
	const Nd4jLong N = 4;

	NDArray a('f', {M,N}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::FLOAT32);
	NDArray x('f', {N}, {1,-2,3,-4}, sd::DataType::DOUBLE);
	NDArray y('f', {M}, sd::DataType::FLOAT32);

	NDArray exp('f', {M}, {0.1, 0.3, 0.5}, sd::DataType::FLOAT32);

	sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
	ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_20) {

	const Nd4jLong M = 3;
	const Nd4jLong N = 4;

	NDArray a('c', {M,N}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::FLOAT32);
	NDArray x('f', {N}, {1,-2,3,-4}, sd::DataType::DOUBLE);
	NDArray y('f', {M}, sd::DataType::FLOAT32);
	NDArray exp('f', {M}, {-1.6, -0.7, 0.2}, sd::DataType::FLOAT32);

	sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
	ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_21) {

	const Nd4jLong M = 3;
	const Nd4jLong N = 4;

	NDArray a('c', {M,N}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::FLOAT32);
	NDArray x('c', {N}, {1,-2,3,-4}, sd::DataType::DOUBLE);
	NDArray y('c', {M}, sd::DataType::FLOAT32);
	NDArray exp('c', {M}, {-1.6, -0.7, 0.2}, sd::DataType::FLOAT32);

	sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
	ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_22) {

	const Nd4jLong M = 3;
	const Nd4jLong N = 4;

	NDArray a('f', {M,N}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::FLOAT32);
	NDArray temp('f', {M,N,5}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
	NDArray x = temp(6, {0,2});
	NDArray y('f', {M}, sd::DataType::FLOAT32);

	NDArray exp('f', {M}, {5.5, 5.1, 4.7}, sd::DataType::FLOAT32);

	sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
	ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_23) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('f', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::FLOAT32);
    a.permutei({1,0});
    NDArray temp('f', {5,M,N}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray x = temp(3, {0,1});
    NDArray y('f', {M}, sd::DataType::FLOAT32);

    NDArray exp('f', {M}, {1.5, 1.8, 1.5}, sd::DataType::FLOAT32);

    sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_24) {

	const Nd4jLong M = 3;
	const Nd4jLong N = 4;

	NDArray a('f', {M,N}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::FLOAT32);
	NDArray temp('f', {M,N,5}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
	NDArray x = temp(6, {0,2},true);
	NDArray y('f', {M}, sd::DataType::FLOAT32);

	NDArray exp('f', {M}, {5.5, 5.1, 4.7}, sd::DataType::FLOAT32);

	sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
	ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_25) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('f', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::FLOAT32);
    a.permutei({1,0});
    NDArray temp('f', {5,M,N}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray x = temp(3, {0,1}, true);
    NDArray y('f', {M}, sd::DataType::FLOAT32);

    NDArray exp('f', {M}, {1.5, 1.8, 1.5}, sd::DataType::FLOAT32);

    sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_26) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('c', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::FLOAT32);
    a.permutei({1,0});
    NDArray temp('f',  {5,M,N}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray temp1('c', {5,M,N}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::FLOAT32);
    NDArray x = temp(2, {0,1});
    NDArray y = temp1(17, {0,2});

    NDArray exp('c', {M}, {-0.3, 0.3, 0.9}, sd::DataType::FLOAT32);

    sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_27) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('c', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::FLOAT32);
    a.permutei({1,0});
    NDArray temp('f',  {5,M,N}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray temp1('c', {5,M,N}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::FLOAT32);
    NDArray x = temp(2, {0,1},true);
    NDArray y = temp1(17, {0,2},true);

    NDArray exp('c', {1,M,1}, {-0.3, 0.3, 0.9}, sd::DataType::FLOAT32);

    sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulMxV_28) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('c', {M,N}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, sd::DataType::FLOAT32);
    NDArray temp('f', {M,N,5}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, sd::DataType::DOUBLE);
    NDArray x = temp(6, {0,2});
    NDArray y('f', {M}, sd::DataType::FLOAT32);

    NDArray exp('f', {M}, {5.1, 3.3, 1.5}, sd::DataType::FLOAT32);

    sd::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulDot_1) {

    const Nd4jLong N = 4;

    NDArray x('c', {N}, {1, 2, 3, 4}, sd::DataType::INT32);
    NDArray y('f', {N}, {0.1, 0.2, 0.3, 0.4}, sd::DataType::FLOAT32);
    NDArray z(sd::DataType::DOUBLE);

    NDArray exp('c', {}, {3}, sd::DataType::DOUBLE);

    sd::MmulHelper::mmul(&x, &y, &z);
    ASSERT_TRUE(z.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulDot_2) {

    const Nd4jLong N = 4;

    NDArray x('c', {1,1,N}, {1,2, 3, 4}, sd::DataType::INT32);
    NDArray y('f', {1,1,N,1,1,1}, {0.1, 0.2, 0.3, 0.4}, sd::DataType::FLOAT32);
    NDArray z(sd::DataType::DOUBLE);

    NDArray exp('c', {}, {3}, sd::DataType::DOUBLE);

    sd::MmulHelper::mmul(&x, &y, &z);
    ASSERT_TRUE(z.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulDot_3) {

    const Nd4jLong N = 4;

    NDArray xBig('c', {4,2}, {1, 0, 2, 0, 3, 0, 4, 0}, sd::DataType::INT32);
    NDArray yBig('c', {4,3}, {0.1, 0, 0, 0.2, 0, 0, 0.3, 0, 0, 0.4, 0,0}, sd::DataType::FLOAT32);
    NDArray x = xBig(0, {1}, true);
    NDArray y = yBig(0, {1}, true);
    NDArray z(sd::DataType::DOUBLE);

    NDArray exp('c', {}, {3}, sd::DataType::DOUBLE);

    sd::MmulHelper::mmul(&x, &y, &z);
    ASSERT_TRUE(z.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(CudaBasicsTests2, mmulDot_4) {

    const Nd4jLong N = 4;

    NDArray xBig('f', {4,2}, {1, 2, 3, 4, 0, 0, 0, 0}, sd::DataType::INT32);
    NDArray yBig('c', {4,3}, {0.1, 0, 0, 0.2, 0, 0, 0.3, 0, 0, 0.4, 0,0}, sd::DataType::FLOAT32);
    NDArray x = xBig(0, {1}, true);
    NDArray y = yBig(0, {1});
    NDArray z(sd::DataType::DOUBLE);

    NDArray exp('c', {}, {3}, sd::DataType::DOUBLE);

    sd::MmulHelper::mmul(&x, &y, &z);
    ASSERT_TRUE(z.equalsTo(&exp));
}
 */