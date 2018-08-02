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
// implementation of operations for Simple Recurrent Unit: arXiv:1709.02755v2 [cs.CL] 12 Sep 2017
//
//@author Yurii Shyrma
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_sru)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/sru.h>
#include <MmulHelper.h>

namespace nd4j {
namespace ops  {

// return 2d array evaluated though last dimension interval t1-t2
template <typename T>
NDArray<T>* timestep(const NDArray<T>* const arr, const int t1, const int t2) {

        IndicesList list({ NDIndex::all(), NDIndex::all(), NDIndex::interval(t1,t2)});
        NDArray<T>* result = arr->subarray(list);     
        result->reshapei(result->ordering(), {arr->shapeOf()[0], arr->shapeOf()[1]} );

        return result;
}

template <typename T>
NDArray<T> _sigmoid(const NDArray<T>& arr) {
    NDArray<T> result(arr.getShapeInfo(), arr.getWorkspace());
    (const_cast<NDArray<T>&>(arr)).template applyTransform<simdOps::Sigmoid<T>>(&result);

    return result;
}


/////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sru_logic, 5, 2, false, 0, 0) {

    NDArray<T>* input   = INPUT_VARIABLE(0);                // X, input 3d tensor [bS x K x N], N - number of time steps, bS - batch size, K - number of features
    NDArray<T>* weights = INPUT_VARIABLE(1);                // W, 2d tensor of weights [3K x K]
    NDArray<T>* bias    = INPUT_VARIABLE(2);                // B, row of biases with twice length [1 × 2*K]
    NDArray<T>* init    = INPUT_VARIABLE(3);                // C_{0}, 2d tensor of initial state [bS x K] at time t=0
    NDArray<T>* mask    = nullptr;                          // optional,  2d tensor of dropout mask [bS x K]

    bool applyMask = false;        
    if (block.width() > 4) {
        mask = INPUT_VARIABLE(4);   
        applyMask = true;
    }

    NDArray<T>* output = OUTPUT_VARIABLE(0);                // h_t, [bS x K x N]
    NDArray<T>* state  = OUTPUT_VARIABLE(1);                // c_t, [bS x K x N]
    
    const int bS     = input->shapeOf()[0];                     // bS - batch size
    const int K      = input->shapeOf()[1];                     // K - number of features
    const int N      = input->shapeOf()[2];                     // N - number of time steps
        
    const NDArray<T> wi = mmul(*weights, *input);                    //  U [bS x 3K x N]            
    const NDArray<T> bF = (*bias)({ {}, {0,  K} });                       // biases for forget gate [1 x K]
    const NDArray<T> bR = (*bias)({ {}, {K,2*K} });                       // biases for reset  gate [1 x K]    

    NDArray<T> xt(block.getWorkspace());
    NDArray<T> zt(block.getWorkspace()); 
    NDArray<T> ft(block.getWorkspace()); 
    NDArray<T> rt(block.getWorkspace());     
    NDArray<T> ht(block.getWorkspace());
    NDArray<T> ct = *init;
    NDArray<T> gct(state->ordering(), {bS, K}, block.getWorkspace());
    NDArray<T> xmt = *input; 
    //  input = input * mask
    if(applyMask)
        xmt.template applyBroadcast<simdOps::Multiply<T>>({0, 1}, mask, &xmt, nullptr);            
    
    for (int t = 0; t < N; ++t) {
  
        xt = xmt({ {}, {},        {t,t+1} }); xt.reshapei(xt.ordering(), {bS, K});       // [bS x  K x N] -> [bS x K x 1] -> [bS x K]
        zt =  wi({ {}, {0,    K}, {t,t+1} }); zt.reshapei(zt.ordering(), {bS, K});       // [bS x 3K x N] -> [bS x K x 1] -> [bS x K]
        ft =  wi({ {}, {K,  2*K}, {t,t+1} }); ft.reshapei(ft.ordering(), {bS, K});       // [bS x 3K x N] -> [bS x K x 1] -> [bS x K]
        rt =  wi({ {}, {2*K,3*K}, {t,t+1} }); rt.reshapei(rt.ordering(), {bS, K});       // [bS x 3K x N] -> [bS x K x 1] -> [bS x K]

        ft = _sigmoid(ft + bF);
        rt = _sigmoid(rt + bR);
        ct = ft * (ct - zt) + zt;                
        // TODO T val = (activation_type == 1) ? tanh(cur) : ((activation_type == 2) ? reluf(cur) : cur );
        ct.template applyTransform<simdOps::Tanh<T>>(&gct);        
        ht = rt * (gct - xt) + xt;

        // save results
        output->assign(ht, {{}, {}, {t,t+1}} );
        state->assign (ct, {{}, {}, {t,t+1}} );
    }    
    
    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(sru_logic) {

    auto inShape = inputShape->at(0);   // [bS x K x N]
    int rank = inShape[0];              // = 3
    int size = rank*2 + 4;
    int bS   = inShape[1];
    int K    = inShape[2];
    int N    = inShape[3];
    char order = (char)(inShape[size-1]);

    Nd4jLong* newShapeInfo1 = nullptr;
    Nd4jLong* newShapeInfo2 = nullptr;
    ALLOCATE(newShapeInfo1, block.getWorkspace(), size, Nd4jLong);
    ALLOCATE(newShapeInfo2, block.getWorkspace(), size, Nd4jLong);
    
    newShapeInfo1[0] = rank;        
    newShapeInfo1[1] = bS;
    newShapeInfo1[2] = K;
    newShapeInfo1[3] = N;
    
    shape::updateStrides(newShapeInfo1, order);
    memcpy(newShapeInfo2, newShapeInfo1, shape::shapeInfoByteLength(newShapeInfo1));
    
    return SHAPELIST(newShapeInfo1, newShapeInfo2);
}   


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sru_old, 5, 2, false, 0, 0) {

    NDArray<T>* x   = INPUT_VARIABLE(0);                // X, input 3d tensor [bS x inSize x time], time - number of time steps, bS - batch size, inSize - number of features
    NDArray<T>* w = INPUT_VARIABLE(1);                // W, 2d tensor of weights [3K x inSize]
    NDArray<T>* b    = INPUT_VARIABLE(2);                // B, row of biases with twice length [1 × 2*inSize]
    NDArray<T>* c0    = INPUT_VARIABLE(3);                // C_{0}, 2d tensor of initial state [bS x inSize] at time t=0
    NDArray<T>* mask    = nullptr;                          // optional,  2d tensor of dropout mask [bS x inSize]

    bool applyMask = false;
    if (block.width() > 4) {
        mask = INPUT_VARIABLE(4);
        applyMask = true;
    }

    NDArray<T>* h = OUTPUT_VARIABLE(0);                // h_t, [bS x inSize x time]
    NDArray<T>* state  = OUTPUT_VARIABLE(1);                // c_t, [bS x inSize x time]

    const int bS     = x->shapeOf()[0];                     // bS - batch size
    const int inSize      = x->shapeOf()[1];                     // inSize - number of features
    const int time      = x->shapeOf()[2];                     // time - number of time steps

    // multiplication matrix = matmul(w,x)
    NDArray<T>* wi = MmulHelper<T>::mmul(w, x, nullptr, (T)1., (T)0.);      //       U [bS x 3K x time]
    // wi.printShapeInfo();
    NDArray<T>* wiZ = wi->subarray( { NDIndex::all(), NDIndex::interval(0,inSize),     NDIndex::all() } );       // [bS x inSize x time]
    NDArray<T>* wiF = wi->subarray( { NDIndex::all(), NDIndex::interval(inSize,2*inSize),   NDIndex::all() } );       // forget gate [bS x inSize x time]
    NDArray<T>* wiR = wi->subarray( { NDIndex::all(), NDIndex::interval(2*inSize,3*inSize), NDIndex::all() } );       // reset gate [bS x inSize x time]
    NDArray<T>* bF  = b->subarray( { NDIndex::all(), NDIndex::interval(0,inSize)  } );                        // biases for forget gate [1 x inSize]
    NDArray<T>* bR  = b->subarray( { NDIndex::all(), NDIndex::interval(inSize,2*inSize)} );                        // biases for reset gate [1 x inSize]

    NDArray<T>* xt(nullptr), *zt(nullptr), *ft(nullptr), *rt(nullptr), *ct(nullptr), *ht(nullptr);
    NDArray<T>* ct_1 = c0->dup(c0->ordering());
    NDArray<T>* gct  = new NDArray<T>(state->ordering(), {bS, inSize});
    NDArray<T>* xmt  = x->dup(x->ordering());
    //  x = x * mask
    if(applyMask)
        xmt->template applyBroadcast<simdOps::Multiply<T>>({0, 1}, mask, xmt, nullptr);            // apply mask

    for (int t = 0; t < time; ++t) {
        xt = timestep(xmt, t, t+1);         // [bS x inSize x time] -> [bS x inSize x 1] -> [bS x inSize]
        zt = timestep(wiZ, t, t+1);         // [bS x inSize x time] -> [bS x inSize x 1] -> [bS x inSize]
        ft = timestep(wiF, t, t+1);         // [bS x inSize x time] -> [bS x inSize x 1] -> [bS x inSize]
        rt = timestep(wiR, t, t+1);         // [bS x inSize x time] -> [bS x inSize x 1] -> [bS x inSize]
        ct = timestep(state, t, t+1);       // [bS x inSize x time] -> [bS x inSize x 1] -> [bS x inSize]
        ht = timestep(h, t, t+1);      // [bS x inSize x time] -> [bS x inSize x 1] -> [bS x inSize]

        // ft = sigmoid(ft + bf), rt = sigmoid(rt + bR)
        ft->addRowVector(bF, ft);
        rt->addRowVector(bR, rt);
        ft->template applyTransform<simdOps::Sigmoid<T>>();
        rt->template applyTransform<simdOps::Sigmoid<T>>();
        // ct = ft * c_t-1 + (1 - ft) * zt,
        ft->template applyPairwiseTransform<simdOps::Multiply<T>>(ct_1, ct, nullptr);
        ft->template applyTransform<simdOps::OneMinus<T>>(ft);
        ft->template applyPairwiseTransform<simdOps::Multiply<T>>(zt, nullptr);
        ct->template applyPairwiseTransform<simdOps::Add<T>>(ft, nullptr);
        // TODO T val = (activation_type == 1) ? tanh(cur) : ((activation_type == 2) ? reluf(cur) : cur );
        ct->template applyTransform<simdOps::Tanh<T>>(gct);

        // ht = rt * gct + (1 - rt) * xt
        rt->template applyPairwiseTransform<simdOps::Multiply<T>>(gct, ht, nullptr);
        rt->template applyTransform<simdOps::OneMinus<T>>(rt);
        rt->template applyPairwiseTransform<simdOps::Multiply<T>>(xt, nullptr);
        ht->template applyPairwiseTransform<simdOps::Add<T>>(rt, nullptr);

        delete xt; delete zt; delete ft; delete rt; delete ht; delete ct_1;
        ct_1 = ct;
    }

    delete wiZ; delete wiF; delete wiR; delete wi; delete bF; delete bR; delete ct_1; delete gct; delete xmt;

    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(sru_old) {

    auto inShape = inputShape->at(0);   // [bS x inSize x time]
    int rank = inShape[0];              // = 3
    int size = rank*2 + 4;
    auto bS   = inShape[1];
    auto inSize    = inShape[2];
    int time    = inShape[3];
    char order = (char)(inShape[size-1]);

    Nd4jLong* newShapeInfo1 = nullptr;
    Nd4jLong* newShapeInfo2 = nullptr;
    ALLOCATE(newShapeInfo1, block.getWorkspace(), size, Nd4jLong);
    ALLOCATE(newShapeInfo2, block.getWorkspace(), size, Nd4jLong);

    newShapeInfo1[0] = rank;
    newShapeInfo1[1] = bS;
    newShapeInfo1[2] = inSize;
    newShapeInfo1[3] = time;

    shape::updateStrides(newShapeInfo1, order);
    memcpy(newShapeInfo2, newShapeInfo1, shape::shapeInfoByteLength(newShapeInfo1));

    return SHAPELIST(newShapeInfo1, newShapeInfo2);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sru, 5, 2, false, 0, 0) {

    NDArray<T>* x    = INPUT_VARIABLE(0);                                   // X, input 3d tensor [bS x inSize x time], time - number of time steps, bS - batch size, inSize - number of features
    NDArray<T>* w    = INPUT_VARIABLE(1);                                   // W, 2d tensor of weights [3*inSize x inSize]
    NDArray<T>* b    = INPUT_VARIABLE(2);                                   // B, row of biases with twice length [2*inSize]
    NDArray<T>* c0   = INPUT_VARIABLE(3);                                   // C_{0}, 2d tensor of initial state [bS x inSize] at time t=0
    NDArray<T>* mask = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;     // optional,  2d tensor of dropout mask [bS x inSize]

    NDArray<T>* h = OUTPUT_VARIABLE(0);                                     // cell outputs, [bS x inSize x time]
    NDArray<T>* c = OUTPUT_VARIABLE(1);                                     // cell states,  [bS x inSize x time]

    const int rank   = x->rankOf();              // = 3
    const int bS     = x->sizeAt(0);
    const int inSize = x->sizeAt(1);
    const int time   = x->sizeAt(2);

    // input shapes validation
    REQUIRE_TRUE(w->rankOf()  == rank-1, 0, "SRU operation: wrong rank of weights array, expected is %i, but got %i instead !", rank-1, w->rankOf());
    REQUIRE_TRUE(b->rankOf()  == 1,      0, "SRU operation: wrong rank of biases  array, expected is %i, but got %i instead !", 1, b->rankOf());
    REQUIRE_TRUE(c0->rankOf() == rank-1, 0, "SRU operation: wrong rank of initial state array, expected is %i, but got %i instead !", rank-1, c0->rankOf());
    if(mask)
        REQUIRE_TRUE(mask->rankOf() == rank-1, 0, "SRU operation: wrong rank of mask array, expected is %i, but got %i instead !", rank-1, mask->rankOf());

    const std::string wShape         = ShapeUtils<T>::shapeAsString(w);
    const std::string wCorrectShape  = ShapeUtils<T>::shapeAsString({3*inSize, inSize});
    const std::string bShape         = ShapeUtils<T>::shapeAsString(b);
    const std::string bCorrectShape  = ShapeUtils<T>::shapeAsString({2*inSize});
    const std::string c0Shape        = ShapeUtils<T>::shapeAsString(c0);
    const std::string c0CorrectShape = ShapeUtils<T>::shapeAsString({bS, inSize});

    REQUIRE_TRUE(wShape  == wCorrectShape,  0, "SRU operation: wrong shape of weights array, expected is %s, but got %s instead !", wCorrectShape.c_str(), wShape.c_str());
    REQUIRE_TRUE(bShape  == bCorrectShape,  0, "SRU operation: wrong shape of biases  array, expected is %s, but got %s instead !", bCorrectShape.c_str(), bShape.c_str());
    REQUIRE_TRUE(c0Shape == c0CorrectShape, 0, "SRU operation: wrong shape of initial state array, expected is %s, but got %s instead !", c0CorrectShape.c_str(), c0Shape.c_str());
    if(mask) {
        const std::string maskShape         = ShapeUtils<T>::shapeAsString(mask);
        REQUIRE_TRUE(maskShape == c0CorrectShape, 0, "SRU operation: wrong shape of mask array, expected is %s, but got %s instead !", c0CorrectShape.c_str(), maskShape.c_str());
    }

    //  xm = x * mask
    NDArray<T>* xm = x;
    if(mask) {
        xm = new NDArray<T>(x->getShapeInfo(), true, block.getWorkspace());
        x->template applyBroadcast<simdOps::Multiply<T>>({0, 1}, mask, xm, nullptr);
    }

    // time loop
    helpers::sruTimeLoop<T>({xm, c0, w, b}, {h, c});

    if(mask)
        delete xm;

    return Status::OK();
}

DECLARE_SHAPE_FN(sru) {

    auto xShapeInfo    = inputShape->at(0);                                   // X, input 3d tensor [bS x inSize x time], time - number of time steps, bS - batch size, inSize - number of features
    auto wShapeInfo    = inputShape->at(1);                                   // W, 2d tensor of weights [3*inSize x inSize]
    auto bShapeInfo    = inputShape->at(2);                                   // B, row of biases with twice length [2*inSize]
    auto c0ShapeInfo   = inputShape->at(3);                                   // C_{0}, 2d tensor of initial state [bS x inSize] at time t=0
    Nd4jLong* maskShapeInfo = block.width() > 4 ? inputShape->at(4) : nullptr;     // optional,  2d tensor of dropout mask [bS x inSize]

    const int rank   = xShapeInfo[0];              // = 3
    const int bS     = xShapeInfo[1];
    const int inSize = xShapeInfo[2];
    const int time   = xShapeInfo[3];

    // input shapes validation
    REQUIRE_TRUE(wShapeInfo[0]  == rank-1, 0, "SRU operation: wrong rank of weights array, expected is %i, but got %i instead !", rank-1, wShapeInfo[0]);
    REQUIRE_TRUE(bShapeInfo[0]  == 1,      0, "SRU operation: wrong rank of biases  array, expected is %i, but got %i instead !", 1, bShapeInfo[0]);
    REQUIRE_TRUE(c0ShapeInfo[0] == rank-1, 0, "SRU operation: wrong rank of initial state array, expected is %i, but got %i instead !", rank-1, c0ShapeInfo[0]);
    if(maskShapeInfo)
        REQUIRE_TRUE(maskShapeInfo[0] == rank-1, 0, "SRU operation: wrong rank of mask array, expected is %i, but got %i instead !", rank-1, maskShapeInfo[0]);

    const std::string wShape         = ShapeUtils<T>::shapeAsString(wShapeInfo);
    const std::string wCorrectShape  = ShapeUtils<T>::shapeAsString({3*inSize, inSize});
    const std::string bShape         = ShapeUtils<T>::shapeAsString(bShapeInfo);
    const std::string bCorrectShape  = ShapeUtils<T>::shapeAsString({2*inSize});
    const std::string c0Shape        = ShapeUtils<T>::shapeAsString(c0ShapeInfo);
    const std::string c0CorrectShape = ShapeUtils<T>::shapeAsString({bS, inSize});

    REQUIRE_TRUE(wShape  == wCorrectShape,  0, "SRU operation: wrong shape of weights array, expected is %s, but got %s instead !", wCorrectShape.c_str(), wShape.c_str());
    REQUIRE_TRUE(bShape  == bCorrectShape,  0, "SRU operation: wrong shape of biases  array, expected is %s, but got %s instead !", bCorrectShape.c_str(), bShape.c_str());
    REQUIRE_TRUE(c0Shape == c0CorrectShape, 0, "SRU operation: wrong shape of initial state array, expected is %s, but got %s instead !", c0CorrectShape.c_str(), c0Shape.c_str());
    if(maskShapeInfo) {
        const std::string maskShape = ShapeUtils<T>::shapeAsString(maskShapeInfo);
        REQUIRE_TRUE(maskShape == c0CorrectShape, 0, "SRU operation: wrong shape of mask array, expected is %s, but got %s instead !", c0CorrectShape.c_str(), maskShape.c_str());
    }

    Nd4jLong* newShapeInfo1 = nullptr;
    Nd4jLong* newShapeInfo2 = nullptr;
    ALLOCATE(newShapeInfo1, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);       // [bS x inSize x time]
    ALLOCATE(newShapeInfo2, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);       // [bS x inSize x time]

    newShapeInfo1[0] = rank;
    newShapeInfo1[1] = bS;
    newShapeInfo1[2] = inSize;
    newShapeInfo1[3] = time;

    shape::updateStrides(newShapeInfo1, shape::order(xShapeInfo));
    memcpy(newShapeInfo2, newShapeInfo1, shape::shapeInfoByteLength(newShapeInfo1));

    return SHAPELIST(newShapeInfo1, newShapeInfo2);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sru_bp, 8, 4, true, 0, 0) {
    
    NDArray<T>* x        = INPUT_VARIABLE(0);                // X, input 3d tensor [bS x K x N], N - number of time steps, bS - batch size, K - number of features
    NDArray<T>* w        = INPUT_VARIABLE(1);                // W, 2d tensor of weights [3K x K]
    NDArray<T>* b        = INPUT_VARIABLE(2);                // B, row of biases with twice length [1 × 2*K]
    NDArray<T>* c0       = INPUT_VARIABLE(3);                // C_{0}, 2d tensor of initial state [bS x K] at time t=0
    NDArray<T>* c        = INPUT_VARIABLE(4);                // C, [bS x K x N]
    NDArray<T>* inGradCt = INPUT_VARIABLE(5);                // [bS x K]
    NDArray<T>* inGradH  = INPUT_VARIABLE(6);                // [bS x K x N]
    NDArray<T>* mask     = nullptr;                          // optional,  2d tensor of dropout mask [bS x K]

    bool applyMask = false;        
    if (block.width() > 7) {
        mask = INPUT_VARIABLE(7);   
        applyMask = true;
    }

    NDArray<T>* gradX    = OUTPUT_VARIABLE(0);              // [bS x K x N]
    NDArray<T>* gradW    = OUTPUT_VARIABLE(1);              // [bS x 3K x K]
    NDArray<T>* gradB    = OUTPUT_VARIABLE(2);              // [1 x 2K]
    NDArray<T>* gradInit = OUTPUT_VARIABLE(3);              // [bS x K]

    const int bS      = x->shapeOf()[0];
    const int K       = x->shapeOf()[1];
    const int N       = x->shapeOf()[2];                     // N - number of time steps
    
    NDArray<T>* gradBias = new NDArray<T>(x->ordering(), {bS, 2*K, N});
    NDArray<T>* gradU    = new NDArray<T>(x->ordering(), {bS, 3*K, N});
    NDArray<T>* gradHX   = new NDArray<T>(x->ordering(), {bS, K, N});
    NDArray<T>* gct      = new NDArray<T>(c->ordering(), {bS, K});
    NDArray<T>* gradTanh = new NDArray<T>(c->ordering(), {bS, K});
    NDArray<T>* gradCt   = new NDArray<T>(c->ordering(), {bS, K});
    NDArray<T>* ftMinus  = new NDArray<T>(c->ordering(), {bS, K});
    NDArray<T>* rtMinus  = new NDArray<T>(c->ordering(), {bS, K});
    NDArray<T>* temp1    = new NDArray<T>(c->ordering(), {bS, K});
    NDArray<T>* temp2    = new NDArray<T>(c->ordering(), {bS, K});

    //  x = x * mask
    if(applyMask)
        x->template applyBroadcast<simdOps::Multiply<T>>({0, 1}, mask, x, nullptr);            // apply mask
    // multiplication matrix wi = matmul(w,x), U = WX
    NDArray<T>* wi = MmulHelper<T>::mmul(w, x, nullptr, (T)1., (T)0.);      // U [bS x 3K x N]

    NDArray<T>* wiZ = wi->subarray( { NDIndex::all(), NDIndex::interval(0,K),     NDIndex::all() } );       // [bS x K x N]
    NDArray<T>* wiF = wi->subarray( { NDIndex::all(), NDIndex::interval(K,2*K),   NDIndex::all() } );       // forget gate [bS x K x N]
    NDArray<T>* wiR = wi->subarray( { NDIndex::all(), NDIndex::interval(2*K,3*K), NDIndex::all() } );       // reset gate [bS x K x N]
    NDArray<T>* bF  = b->subarray( { NDIndex::all(), NDIndex::interval(0,K)  } );                        // biases for forget gate [1 x K]
    NDArray<T>* bR  = b->subarray( { NDIndex::all(), NDIndex::interval(K,2*K)} );                        // biases for reset gate [1 x K]
    NDArray<T>* gradBF = gradBias->subarray( { NDIndex::all(), NDIndex::interval(0,K),   NDIndex::all() } );   // [bS x K x N]
    NDArray<T>* gradBR = gradBias->subarray( { NDIndex::all(), NDIndex::interval(K,2*K), NDIndex::all() } );   // [bS x K x N]
    NDArray<T>* gradUZ = gradU->subarray( { NDIndex::all(), NDIndex::interval(0,K),     NDIndex::all() } ); // [bS x K x N]
    NDArray<T>* gradUF = gradU->subarray( { NDIndex::all(), NDIndex::interval(K,2*K),   NDIndex::all() } ); // [bS x K x N]
    NDArray<T>* gradUR = gradU->subarray( { NDIndex::all(), NDIndex::interval(2*K,3*K), NDIndex::all() } ); // [bS x K x N]


    NDArray<T>* xt(nullptr), *zt(nullptr), *ft(nullptr), *rt(nullptr), *ct(nullptr), *inGradHt(nullptr), *gradBFt(nullptr), 
                *gradBRt(nullptr), *ct_1(nullptr), *gradHXt(nullptr), *gradURt(nullptr), *gradUFt(nullptr), *gradUZt(nullptr);

    for (int t = N-1; t >=0 ; --t) {           
        // initialization
        xt = timestep(x, t, t+1);               // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        zt = timestep(wiZ, t, t+1);                 // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        ft = timestep(wiF, t, t+1);                 // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        rt = timestep(wiR, t, t+1);                 // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        ct = timestep(c, t, t+1);               // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        inGradHt = timestep(inGradH, t, t+1);       // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        gradBRt  = timestep(gradBR, t, t+1);        // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        gradBFt  = timestep(gradBF, t, t+1);        // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        gradHXt  = timestep(gradHX, t, t+1);        // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        gradUZt  = timestep(gradUZ, t, t+1);        // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        gradUFt  = timestep(gradUF, t, t+1);        // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        gradURt  = timestep(gradUR, t, t+1);        // [bS x K x N] -> [bS x K x 1] -> [bS x K]                        

        if(t != 0)
            ct_1  = timestep(c, t-1, t);        // previous c_{t-1}
        else
            ct_1 = c0->dup(c0->ordering());
        
        ///////////////// forward
        // ft = sigmoid(ft + bf), rt = sigmoid(rt + bR)
        ft->addRowVector(bF, ft);
        rt->addRowVector(bR, rt);
        ft->template applyTransform<simdOps::Sigmoid<T>>();
        rt->template applyTransform<simdOps::Sigmoid<T>>();
        
        // TODO T val = (activation_type == 1) ? tanh(cur) : ((activation_type == 2) ? reluf(cur) : cur );
        ct->template applyTransform<simdOps::Tanh<T>>(gct);
        // ftMinus = 1-ft,  rtMinus = 1-rt
        ft->template applyTransform<simdOps::OneMinus<T>>(ftMinus);
        rt->template applyTransform<simdOps::OneMinus<T>>(rtMinus);

        ///////////////// backward
        // bR, *grad_brt_ptr = inGradHt * (g_ct - xt) * (1.0f - rt) * rt;
        gct->template applyPairwiseTransform<simdOps::Subtract<T>>(xt, temp1, nullptr);                 // temp1 = (g_ct - xt)                
        rtMinus->template applyPairwiseTransform<simdOps::Multiply<T>>(rt, temp2, nullptr);             // temp2 = (1.0f - rt) * rt;
        temp1->template applyPairwiseTransform<simdOps::Multiply<T>>(temp2, nullptr);                   // temp1 = (g_ct - xt) * (1.0f - rt) * rt;
        inGradHt->template applyPairwiseTransform<simdOps::Multiply<T>>(temp1, gradBRt, nullptr);       // = inGradHt * (g_ct - xt) * (1.0f - rt) * rt;
        
        // bF, TODO - tanh
        // gradTanh = (1.0f - g_ct * g_ct);
        gct->template applyPairwiseTransform<simdOps::Multiply<T>>(gct, gradTanh, nullptr);             // gradTanh = g_ct * g_ct
        gradTanh->template applyTransform<simdOps::OneMinus<T>>(gradTanh);                              // gradTanh = (1.0f - g_ct * g_ct)
        // gradCt  = inGradHt * rt * gradTanh                
        rt->template applyPairwiseTransform<simdOps::Multiply<T>>(gradTanh, gradCt, nullptr);           // gradCt = rt * gradTanh
        inGradHt->template applyPairwiseTransform<simdOps::Multiply<T>>(gradCt, gradCt, nullptr);       // gradCt = inGradHt * rt * gradTanh        
        // gradBFt = (gradCt + inGradCt) * (ct_1 - zt) * (1 - ft) * ft;
        gradCt->template applyPairwiseTransform<simdOps::Add<T>>(inGradCt, temp1, nullptr);              // temp1 = (gradCt + inGradCt)
        ct_1->template applyPairwiseTransform<simdOps::Subtract<T>>(zt, temp2, nullptr);                // temp2 = (ct_1 - zt)
        temp1->template applyPairwiseTransform<simdOps::Multiply<T>>(ftMinus, temp1, nullptr);          // temp1 = (gradCt + inGradCt)*(1-ft)
        temp1->template applyPairwiseTransform<simdOps::Multiply<T>>(ft, temp1, nullptr);               // temp1 = (gradCt + inGradCt)*(1-ft)*ft
        temp1->template applyPairwiseTransform<simdOps::Multiply<T>>(temp2, gradBFt, nullptr);          // gradBFt = (gradCt + inGradCt) * (ct_1 - zt) * (1 - ft) * ft;

        // x_t (highway connection), gradHXt = inGradHt * (1.0f - rt);
        inGradHt->template applyPairwiseTransform<simdOps::Multiply<T>>(rtMinus, gradHXt, nullptr);

        // U_t, gradUZt = (inGradHt * rt * grad_tanh + inGradCt) * (1.0f - ft);
        rt->template applyPairwiseTransform<simdOps::Multiply<T>>(gradTanh, temp1, nullptr);        // temp1 = rt * grad_tanh 
        inGradHt->template applyPairwiseTransform<simdOps::Multiply<T>>(temp1, temp1, nullptr);     // temp1 = inGradHt * rt * grad_tanh 
        temp1->template applyPairwiseTransform<simdOps::Add<T>>(inGradCt, temp1, nullptr);          // temp1 = inGradHt * rt * grad_tanh + inGradCt
        temp1->template applyPairwiseTransform<simdOps::Multiply<T>>(ftMinus, gradUZt, nullptr);    // gradUZt = (inGradHt * rt * grad_tanh + inGradCt) * (1.0f - ft);
        gradUFt->assign(gradBFt);
        gradURt->assign(gradBRt);

        // c_{t-1}, inGradCt = (gradCt + inGradCt) * ft;
        gradCt->template applyPairwiseTransform<simdOps::Add<T>>(inGradCt, temp1, nullptr);         // temp1 = (gradCt + inGradCt)
        temp1->template applyPairwiseTransform<simdOps::Multiply<T>>(ft, inGradCt, nullptr);        // inGradCt = (gradCt + inGradCt) * ft;
        
        delete xt; delete zt; delete ft; delete rt; delete ct; delete inGradHt; delete ct_1; delete gradBRt; 
        delete gradBFt; delete gradHXt; delete gradUZt; delete gradUFt; delete gradURt;
    }

    // gradInit
    gradInit->assign(inGradCt);

    // gradX 
    NDArray<T>* weightsT = w->transpose();                                            // [K x 3K]
    MmulHelper<T>::mmul(weightsT, gradU, gradX, (T)1., (T)0.);                    // [bS x K x N]    
    gradX->template applyPairwiseTransform<simdOps::Add<T>>(gradHX, gradX, nullptr);        // + grad_highway_x
    if(applyMask)
        gradX->template applyBroadcast<simdOps::Multiply<T>>({0,1}, mask, gradX, nullptr);  // apply mask

    // gradB    
    NDArray<T>* temp3 = gradBias->template reduceAlongDimension<simdOps::Sum<T>>({0,2}, false, true);    // [1 x 2K]
    gradB->assign(temp3);

    // gradW [bS x 3K x K]
    x->permutei({0, 2, 1});                                               // [bS x N x K]
    MmulHelper<T>::mmul(gradU, x, gradW, (T)1., (T)0.);          // [bS x 3K x K]

    delete gradUR; delete gradBF; delete gradUZ; delete gradUF; delete gradBR;

    delete gct;   delete gradU; delete gradHX; delete wiZ; delete wiF; delete wiR; delete bF; delete bR;
    delete temp1; delete temp2; delete temp3; delete gradCt; delete wi;
    delete gradTanh; delete ftMinus; delete rtMinus; delete weightsT; delete gradBias;
    
    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(sru_bp) {

    auto inShape = inputShape->at(0);   // [bS x inSize x time]
    auto bS   = inShape[1];
    auto inSize    = inShape[2];
    auto time    = inShape[3];
    char order = (char)(inShape[9]);

    Nd4jLong *newShapeInfo1(nullptr), *newShapeInfo2(nullptr), *newShapeInfo3(nullptr), *newShapeInfo4(nullptr);
    ALLOCATE(newShapeInfo1, block.getWorkspace(), 10, Nd4jLong);
    ALLOCATE(newShapeInfo2, block.getWorkspace(), 10, Nd4jLong);
    ALLOCATE(newShapeInfo3, block.getWorkspace(), 8, Nd4jLong);
    ALLOCATE(newShapeInfo4, block.getWorkspace(), 8,  Nd4jLong);    
    
    newShapeInfo1[0] = 3;
    newShapeInfo1[1] = bS;
    newShapeInfo1[2] = inSize;
    newShapeInfo1[3] = time;
    shape::updateStrides(newShapeInfo1, order);

    newShapeInfo2[0] = 3;        
    newShapeInfo2[1] = bS;
    newShapeInfo2[2] = 3*inSize;
    newShapeInfo2[3] = inSize;
    shape::updateStrides(newShapeInfo2, order);

    newShapeInfo3[0] = 2;
    newShapeInfo3[1] = 1;
    newShapeInfo3[2] = 2*inSize;
    shape::updateStrides(newShapeInfo3, order);

    newShapeInfo4[0] = 2;        
    newShapeInfo4[1] = bS;
    newShapeInfo4[2] = inSize;
    shape::updateStrides(newShapeInfo4, order);
    
    return SHAPELIST(newShapeInfo1, newShapeInfo2, newShapeInfo3, newShapeInfo4);
}   
 

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sru_bp_logic, 8, 4, true, 0, 0) {
    
    NDArray<T>* x        = INPUT_VARIABLE(0);                                   // X, input 3d tensor [bS x inSize x time], time - number of time steps, bS - batch size, inSize - number of features
    NDArray<T>* w        = INPUT_VARIABLE(1);                                   // W, 2d tensor of weights [3*inSize x inSize]
    NDArray<T>* b        = INPUT_VARIABLE(2);                                   // B, row of biases with twice length [1 × 2*inSize]
    NDArray<T>* c0       = INPUT_VARIABLE(3);                                   // C_{0}, 2d tensor of initial state [bS x inSize] at time t=0
    NDArray<T>* c        = INPUT_VARIABLE(4);                                   // C, [bS x inSize x time]
    NDArray<T>* inGradCt = INPUT_VARIABLE(5);                                   // [bS x inSize]
    NDArray<T>* inGradH  = INPUT_VARIABLE(6);                                   // [bS x inSize x time]
    NDArray<T>* mask     = block.width() > 7 ? INPUT_VARIABLE(7) : nullptr;     // optional,  2d tensor of dropout mask [bS x inSize]

    NDArray<T>* gradX    = OUTPUT_VARIABLE(0);              // [bS x inSize x time]
    NDArray<T>* gradW    = OUTPUT_VARIABLE(1);              // [bS x 3*inSize x inSize]
    NDArray<T>* gradB    = OUTPUT_VARIABLE(2);              // [2*inSize]
    NDArray<T>* gradInit = OUTPUT_VARIABLE(3);              // [bS x inSize]

    // input shapes validation
    const int rank = 3;
    REQUIRE_TRUE(x->rankOf()  == rank,   0, "SRU_BP operation: wrong rank of input array, expected is %i, but got %i instead !", rank, x->rankOf());
    REQUIRE_TRUE(w->rankOf()  == rank-1, 0, "SRU_BP operation: wrong rank of weights array, expected is %i, but got %i instead !", rank-1, w->rankOf());
    REQUIRE_TRUE(b->rankOf()  <= 2,      0, "SRU_BP operation: wrong rank of biases  array, expected is <=2, but got %i instead !", b->rankOf());
    REQUIRE_TRUE(c0->rankOf() == rank-1, 0, "SRU_BP operation: wrong rank of initial state array, expected is %i, but got %i instead !", rank-1, c0->rankOf());
    REQUIRE_TRUE(c->rankOf()  == rank,   0, "SRU_BP operation: wrong rank of cell states array, expected is %i, but got %i instead !", rank, c->rankOf());
    REQUIRE_TRUE(inGradCt->rankOf() == rank-1, 0, "SRU_BP operation: wrong rank of array of cell state gradient, expected is %i, but got %i instead !", rank-1, inGradCt->rankOf());
    REQUIRE_TRUE(inGradH->rankOf()  == rank,   0, "SRU_BP operation: wrong rank of array of cell outputs gradients, expected is %i, but got %i instead !", rank, inGradH->rankOf());
    if(mask)
        REQUIRE_TRUE(mask->rankOf() == rank-1, 0, "SRU_BP operation: wrong rank of mask array, expected is %i, but got %i instead !", rank-1, mask->rankOf());

    const int bS      = x->shapeOf()[0];
    const int inSize  = x->shapeOf()[1];
    const int time    = x->shapeOf()[2];                     // time - number of time steps

    const std::string wShape               = ShapeUtils<T>::shapeAsString(w);
    const std::string wCorrectShape        = ShapeUtils<T>::shapeAsString({3*inSize, inSize});
    // const std::string bShape               = ShapeUtils<T>::shapeAsString(b);
    // const std::string bCorrectShape        = ShapeUtils<T>::shapeAsString({2*inSize});
    const std::string c0Shape              = ShapeUtils<T>::shapeAsString(c0);
    const std::string c0CorrectShape       = ShapeUtils<T>::shapeAsString({bS, inSize});
    const std::string cShape               = ShapeUtils<T>::shapeAsString(c);
    const std::string cCorrectShape        = ShapeUtils<T>::shapeAsString({bS, inSize, time});
    const std::string inGradCtShape        = ShapeUtils<T>::shapeAsString(inGradCt);
    const std::string inGradCtCorrectShape = ShapeUtils<T>::shapeAsString({bS, inSize});
    const std::string inGradHShape         = ShapeUtils<T>::shapeAsString(inGradH);
    const std::string inGradHCorrectShape  = ShapeUtils<T>::shapeAsString({bS, inSize, time});
    
    REQUIRE_TRUE(wShape  == wCorrectShape,  0, "SRU_BP operation: wrong shape of weights array, expected is %s, but got %s instead !", wCorrectShape.c_str(), wShape.c_str());
    // REQUIRE_TRUE(bShape  == bCorrectShape,  0, "SRU_BP operation: wrong shape of biases  array, expected is %s, but got %s instead !", bCorrectShape.c_str(), bShape.c_str());
    REQUIRE_TRUE(c0Shape == c0CorrectShape, 0, "SRU_BP operation: wrong shape of initial state array, expected is %s, but got %s instead !", c0CorrectShape.c_str(), c0Shape.c_str());
    REQUIRE_TRUE(cShape == cCorrectShape, 0, "SRU_BP operation: wrong shape of cell states array, expected is %s, but got %s instead !", cCorrectShape.c_str(), cShape.c_str());
    REQUIRE_TRUE(inGradCtShape == inGradCtCorrectShape, 0, "SRU_BP operation: wrong shape of array of cell state gradient, expected is %s, but got %s instead !", inGradCtCorrectShape.c_str(), inGradCtShape.c_str());
    REQUIRE_TRUE(inGradHShape == inGradHCorrectShape, 0, "SRU_BP operation: wrong shape of array of cell outputs gradients, expected is %s, but got %s instead !", inGradHCorrectShape.c_str(), inGradHShape.c_str());
    if(mask) {
        const std::string maskShape = ShapeUtils<T>::shapeAsString(mask);
        REQUIRE_TRUE(maskShape == c0CorrectShape, 0, "SRU_BP operation: wrong shape of mask array, expected is %s, but got %s instead !", c0CorrectShape.c_str(), maskShape.c_str());
    }


    const NDArray<T> bF = (*b)({ {}, {0,  inSize} });                                 // biases for forget gate [1 x inSize]
    const NDArray<T> bR = (*b)({ {}, {inSize,2*inSize} });                                 // biases for reset  gate [1 x inSize]
    NDArray<T> gradBias(x->ordering(),   {bS, 2*inSize, time}, block.getWorkspace());
    NDArray<T> gradU   (x->ordering(),   {bS, 3*inSize, time}, block.getWorkspace());
    NDArray<T> gradHX  (x->ordering(),   {bS,   inSize, time}, block.getWorkspace());
    NDArray<T> gct     (c->ordering(),   {bS, inSize},      block.getWorkspace());

    //  x = x * mask
    if(mask)
        x->template applyBroadcast<simdOps::Multiply<T>>({0, 1}, mask, x, nullptr);             // apply mask
    // multiplication matrix wi = matmul(w,x), U = WX
    const NDArray<T> wi = mmul(*w, *x);                                                   //  U [bS x 3K x time]

    for (int t = time-1; t >=0 ; --t) {
        // initialization
        NDArray<T> xt =         (*x)({ {}, {},        {t,t+1} }); xt.reshapei(xt.ordering(), {bS, inSize});          // [bS x inSize  x time] -> [bS x inSize x 1] -> [bS x inSize]
        NDArray<T> zt =               wi({ {}, {0,    inSize}, {t,t+1} }); zt.reshapei(zt.ordering(), {bS, inSize});          // [bS x 3K x time] -> [bS x inSize x 1] -> [bS x inSize]
        NDArray<T> ft =               wi({ {}, {inSize,  2*inSize}, {t,t+1} }); ft.reshapei(ft.ordering(), {bS, inSize});          // [bS x 3K x time] -> [bS x inSize x 1] -> [bS x inSize]
        NDArray<T> rt =               wi({ {}, {2*inSize,3*inSize}, {t,t+1} }); rt.reshapei(rt.ordering(), {bS, inSize});          // [bS x 3K x time] -> [bS x inSize x 1] -> [bS x inSize]
        NDArray<T> ct =         (*c)({ {}, {},        {t,t+1} }); ct.reshapei(ct.ordering(), {bS, inSize});          // [bS x inSize  x time] -> [bS x inSize x 1] -> [bS x inSize]
        NDArray<T> inGradHt = (*inGradH)({ {}, {},        {t,t+1} }); inGradHt.reshapei(xt.ordering(), {bS, inSize});    // [bS x inSize  x time] -> [bS x inSize x 1] -> [bS x inSize]

        NDArray<T> ct_1 = t ? (*c)({ {}, {}, {t-1,t} }) : *c0;                                                // previous c_{t-1}
        
        ///////////////// forward
        // ft = sigmoid(ft + bf), rt = sigmoid(rt + bR)
        ft = _sigmoid(ft + bF);
        rt = _sigmoid(rt + bR);        
        // TODO T val = (activation_type == 1) ? tanh(cur) : ((activation_type == 2) ? reluf(cur) : cur );
        ct.template applyTransform<simdOps::Tanh<T>>(&gct);        

        ///////////////// backward
        // bR, *grad_brt_ptr = inGradHt * (g_ct - xt) * (1.0f - rt) * rt;
        // ftMinus = -ft + (T)1.;
        NDArray<T> ftMinus = (T)1. - ft;
        NDArray<T> rtMinus = (T)1. - rt;
        NDArray<T> gradBRt = inGradHt * (gct - xt) * rtMinus * rt;
        // bF, TODO - tanh            
        NDArray<T> gradTanh = (T)1. - gct * gct;
        NDArray<T> gradCt = inGradHt * rt * gradTanh;
        NDArray<T> gradBFt = (gradCt + *inGradCt) * (ct_1 - zt) * ftMinus * ft;
        // x_t (highway connection), gradHXt = inGradHt * (1.0f - rt);
        NDArray<T> gradHXt = inGradHt * rtMinus;

        // U_t, gradUZt = (inGradHt * rt * grad_tanh + inGradCt) * (1.0f - ft);
        NDArray<T> gradUZt = (inGradHt * rt * gradTanh + *inGradCt) * ftMinus;

        // c_{t-1}, inGradCt = (gradCt + inGradCt) * ft;
        *inGradCt = (gradCt + *inGradCt) * ft;
        
        // save results        
        gradBias.assign(gradBFt, {{}, {0,     inSize}, {t,t+1}} );
        gradBias.assign(gradBRt, {{}, {inSize,   2*inSize}, {t,t+1}} );
        gradU.assign(gradUZt,    {{}, {0,     inSize}, {t,t+1}} );
        gradU.assign(gradBFt,    {{}, {inSize,   2*inSize}, {t,t+1}} );
        gradU.assign(gradBRt,    {{}, {2*inSize, 3*inSize}, {t,t+1}} );
        gradHX.assign(gradHXt,   {{}, {        }, {t,t+1}} );              
    }

    // gradInit
    gradInit->assign(inGradCt);
    // gradX 
    w->transposei();                                                               // [inSize x 3K]
    gradX->assign( mmul(*w, gradU) + gradHX);
    if(mask)
        gradX->template applyBroadcast<simdOps::Multiply<T>>({0,1}, mask, gradX, nullptr);       // apply mask

    // gradB    
    gradBias.template reduceAlongDimension<simdOps::Sum<T>>(gradB, {0,2}, false, true);    // [1 x 2K]    

    // gradW [bS x 3K x inSize]
    x->permutei({0, 2, 1});                                               // [bS x time x inSize]
    gradW->assign( mmul(gradU, *x) );
    
    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(sru_bp_logic) {

    auto inShape = inputShape->at(0);   // [bS x inSize x time]
    auto bS   = inShape[1];
    auto inSize    = inShape[2];
    auto time    = inShape[3];
    char order = shape::order(inShape);

    Nd4jLong *newShapeInfo1(nullptr), *newShapeInfo2(nullptr), *newShapeInfo3(nullptr), *newShapeInfo4(nullptr);
    ALLOCATE(newShapeInfo1, block.getWorkspace(), 10, Nd4jLong);
    ALLOCATE(newShapeInfo2, block.getWorkspace(), 10, Nd4jLong);
    ALLOCATE(newShapeInfo3, block.getWorkspace(), 8, Nd4jLong);
    ALLOCATE(newShapeInfo4, block.getWorkspace(), 8,  Nd4jLong);    
    
    newShapeInfo1[0] = 3;
    newShapeInfo1[1] = bS;
    newShapeInfo1[2] = inSize;
    newShapeInfo1[3] = time;
    shape::updateStrides(newShapeInfo1, order);

    newShapeInfo2[0] = 3;        
    newShapeInfo2[1] = bS;
    newShapeInfo2[2] = 3*inSize;
    newShapeInfo2[3] = inSize;
    shape::updateStrides(newShapeInfo2, order);

    newShapeInfo3[0] = 2;
    newShapeInfo3[1] = 1;
    newShapeInfo3[2] = 2*inSize;
    shape::updateStrides(newShapeInfo3, order);

    newShapeInfo4[0] = 2;        
    newShapeInfo4[1] = bS;
    newShapeInfo4[2] = inSize;
    shape::updateStrides(newShapeInfo4, order);
    
    return SHAPELIST(newShapeInfo1, newShapeInfo2, newShapeInfo3, newShapeInfo4);
}   

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sru_bi, 5, 2, true, 0, 0) {

    NDArray<T>* x   = INPUT_VARIABLE(0);                // X, input 3d tensor [time x bS x 2K], time - number of time steps, bS - batch size, inSize - number of features
    NDArray<T>* w = INPUT_VARIABLE(1);                // W, 2d tensor of weights [2K x 6K]
    NDArray<T>* b    = INPUT_VARIABLE(2);                // B, row of biases with twice length [1 × 4K]
    NDArray<T>* c0    = INPUT_VARIABLE(3);                // C_{0}, 2d tensor of initial state [bS x 2K] at time t=0
    NDArray<T>* mask    = nullptr;                          // optional, 2d tensor of dropout mask [bS x 2K]

    bool applyMask = false;        
    if (block.width() > 4) {
        mask = INPUT_VARIABLE(4);   
        applyMask = true;
    }

    NDArray<T>* ht = OUTPUT_VARIABLE(0);                // h_t, [time x bS x 2K]
    NDArray<T>* state  = OUTPUT_VARIABLE(1);                // c_t, [time x bS x 2K]
    
    const int time   = x->shapeOf()[0];                     // time - number of time steps
    const int bS     = x->shapeOf()[1];                     // bS - batch size
    const int inSize = x->shapeOf()[2] / 2;                 // inSize - number of features
  
    //  x = x * mask
    if(applyMask)
        x->template applyBroadcast<simdOps::Multiply<T>>({1, 2}, mask, x, nullptr);             // apply mask
    // U = x * w
    NDArray<T> wi = mmul(*x, *w);                    //  U [time x bS x 6K]

    const int d2      = 2*inSize;
    const int ncols   = bS*d2;     
    const int ncolsWi = 3*ncols;    

    T* const pInput  = x->getBuffer();
    T* const pWi     = wi.getBuffer();
    T* const pBias   = b->getBuffer();
    T* const pInit   = c0->getBuffer();
    T* const pMask   = mask->getBuffer();
    T* const pOutput = ht->getBuffer();
    T* const pState  = state->getBuffer();

    int ncolsRev, ncolsWiRev;                   // for reverse direction
    T maskVal, cur, bF, bR, ft, rt, val;
    T *pInputVal(nullptr), *pWiVal(nullptr), *pOutputVal(nullptr), *pStateVal(nullptr);
    bool flip = false;

    for (int col = 0; col < ncols; ++col) {           
        
        flip       = (col%d2) >= inSize;
        maskVal    = applyMask ? *(pMask + col) : (T)1.;
        cur        = *(pInit + col);
        bF         = *(pBias + col%d2);
        bR         = *(pBias + col%d2 + d2);
        pWiVal     = pWi     + 3*col;
        pInputVal  = pInput  + col;
        pOutputVal = pOutput + col;
        pStateVal  = pState  + col;

        if (flip) {
            pInputVal  += (time-1)*ncols;
            pWiVal     += (time-1)*ncolsWi;
            pOutputVal += (time-1)*ncols;
            pStateVal  += (time-1)*ncols;
        }

        ncolsRev   = flip ? -ncols   : ncols;
        ncolsWiRev = flip ? -ncolsWi : ncolsWi;
        
        for (int t = 0; t < time; ++t) {
            // evaluate sigmoids 
            ft = ((T)1.)/((T)1. + nd4j::math::nd4j_exp<T>(-(*(pWiVal + 1) + bF)));
            rt = ((T)1.)/((T)1. + nd4j::math::nd4j_exp<T>(-(*(pWiVal + 2) + bR)));
            
            cur = (cur - *pWiVal)*ft + *pWiVal;
            *pStateVal = cur;
            val = nd4j::math::nd4j_tanh<T>(cur);            
            *pOutputVal = (val*maskVal - *pInputVal)*rt + *pInputVal;

            pInputVal  += ncolsRev;
            pWiVal     += ncolsWiRev;            
            pStateVal  += ncolsRev;
            pOutputVal += ncolsRev;
        } 
    }
    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(sru_bi) {

    auto inShape = inputShape->at(0);   // [time x bS x 2K ]
    auto rank = inShape[0];              // = 3
    auto size = rank*2 + 4;        
    auto time    = inShape[1];
    auto bS   = inShape[2];
    auto inSize    = inShape[3] / 2;
    
    char order = shape::order(inShape);

    Nd4jLong* newShapeInfo1 = nullptr;
    Nd4jLong* newShapeInfo2 = nullptr;
    ALLOCATE(newShapeInfo1, block.getWorkspace(), size, Nd4jLong);
    ALLOCATE(newShapeInfo2, block.getWorkspace(), size, Nd4jLong);
    
    newShapeInfo1[0] = rank;        
    newShapeInfo1[1] = time;
    newShapeInfo1[2] = bS;
    newShapeInfo1[3] = 2*inSize;
    
    // FIXME: remove memcpy
    shape::updateStrides(newShapeInfo1, order);
    memcpy(newShapeInfo2, newShapeInfo1, shape::shapeInfoByteLength(newShapeInfo1));
    
    return SHAPELIST(newShapeInfo1, newShapeInfo2);
}   


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sru_bi_bp, 8, 4, true, 0, 0) {
    
    NDArray<T>* x    = INPUT_VARIABLE(0);                // X, input 3d tensor [time x bS x 2K], time - number of time steps, bS - batch size, inSize - number of features
    NDArray<T>* w  = INPUT_VARIABLE(1);                // W, 2d tensor of weights [2K x 6K]
    NDArray<T>* b     = INPUT_VARIABLE(2);                // B, row of biases with twice length [1 × 4K]
    NDArray<T>* c0     = INPUT_VARIABLE(3);                // C_{0}, 2d tensor of initial state [bS x 2K] at time t=0
    NDArray<T>* state    = INPUT_VARIABLE(4);                // C, [time x bS x 2K]
    NDArray<T>* inGradCt = INPUT_VARIABLE(5);                // [bS x 2K]
    NDArray<T>* inGradH  = INPUT_VARIABLE(6);                // [time x bS x 2K]
    NDArray<T>* mask     = nullptr;                          // optional,  2d tensor of dropout mask [bS x 2K]

    bool applyMask = false;        
    if (block.width() > 7) {
        mask = INPUT_VARIABLE(7);   
        applyMask = true;
    }

    NDArray<T>* gradInput   = OUTPUT_VARIABLE(0);              // [time x bS x 2K]
    NDArray<T>* gradWeights = OUTPUT_VARIABLE(1);              // [time x 2K x 6K]
    NDArray<T>* gradB       = OUTPUT_VARIABLE(2);              // [1 x 4K]
    NDArray<T>* gradInit    = OUTPUT_VARIABLE(3);              // [bS x 2K]

    const int time       = x->shapeOf()[0];                     // time - number of time steps
    const int bS      = x->shapeOf()[1];
    const int inSize       = x->shapeOf()[2] / 2;

    //  x = x * mask
    if(applyMask)
        x->template applyBroadcast<simdOps::Multiply<T>>({1, 2}, mask, x, nullptr);             // apply mask
    // U = x * w
    NDArray<T> wi = mmul(*x, *w);                    //  [time x bS x 2K] * [2K x 6K] = [time x bS x 6K]

    NDArray<T> gradBias(x->ordering(), {bS, 4*inSize},    block.getWorkspace());
    NDArray<T> gradWi  (x->ordering(), {time, bS, 6*inSize}, block.getWorkspace());
    
    const int d2      = 2*inSize;
    const int ncols   = bS*d2;     
    const int ncolsWi = 3*ncols;    

    T* const pInput     = x->getBuffer();
    T* const pWi        = wi.getBuffer();
    T* const pBias      = b->getBuffer();
    T* const pInit      = c0->getBuffer();
    T* const pMask      = mask->getBuffer();    
    T* const pState     = state->getBuffer();
    T* const pInGradCt  = inGradCt->getBuffer();
    T* const pInGradH   = inGradH->getBuffer();
    T* const pGradWi    = gradWi.getBuffer();
    T* const pGradInput = gradInput->getBuffer();
    T* const pGradBias  = gradBias.getBuffer();
    T* const pGradInit  = gradInit->getBuffer();

    int ncolsRev, ncolsWiRev;                   // for reverse direction
    T gbF, gbR, cur, maskVal, bF, bR, ft, rt, val, prevVal, gft, grt, gradSateVal;
    bool flip = false;
    T *pInputVal(nullptr), *pWiVal(nullptr),  *pStateVal(nullptr), *pInGradHVal(nullptr), *pGradWiVal(nullptr), *pGradInputVal(nullptr); 

    for (int col = 0; col < ncols; ++col) {           

        gbF = gbR = (T)0.;

        flip          = (col%d2) >= inSize;
        maskVal       = applyMask ? *(pMask + col) : (T)1.;
        cur           = *(pInGradCt + col);
        bF            = *(pBias     + col%d2);
        bR            = *(pBias     + col%d2 + d2);
        pWiVal        = pWi         + 3*col;
        pInputVal     = pInput      + col;
        pStateVal     = pState      + col;
        pInGradHVal   = pInGradH    + col;
        pGradWiVal    = pGradWi     + 3*col;
        pGradInputVal = pGradInput  + col;                    

        if (!flip) {
            pInputVal     += (time-1)*ncols;
            pWiVal        += (time-1)*ncolsWi;
            pStateVal     += (time-1)*ncols;
            pInGradHVal   += (time-1)*ncols;
            pGradWiVal    += (time-1)*ncolsWi;
            pGradInputVal += (time-1)*ncols;
        }

        ncolsRev   = flip ? -ncols   : ncols;
        ncolsWiRev = flip ? -ncolsWi : ncolsWi;
        
        for (int t = 0; t < time; ++t) {
            // evaluate sigmoids 
            ft = ((T)1.)/((T)1. + nd4j::math::nd4j_exp<T>(-(*(pWiVal + 1) + bF)));
            rt = ((T)1.)/((T)1. + nd4j::math::nd4j_exp<T>(-(*(pWiVal + 2) + bR)));
            
            val     = nd4j::math::nd4j_tanh<T>(*pStateVal);            
            prevVal = (t < time-1) ? (*(pStateVal - ncolsRev)) : (*(pInit + col));
            // grad wrt input
            *pGradInputVal = *pInGradHVal - (*pInGradHVal)*rt ;
            // grad wrt rt, wiR and bR
            grt = (*pInGradHVal) * (val*maskVal - *pInputVal) * (rt - rt*rt);
            *(pGradWiVal + 2) = grt;
            gbR += grt;
            // grad wrt state          
            gradSateVal = (*pInGradHVal) * maskVal * (rt - rt*val*val) + cur;
            // grad wrt wi0
            *pGradWiVal = gradSateVal - gradSateVal*ft;
            // grad wrt ft, wi1, and bF
            gft = gradSateVal * (prevVal - *pWiVal) * (ft - ft*ft);
            *(pGradWiVal + 1) = gft;
            gbF += gft;
            // grad wrt c_previous
            cur = gradSateVal * ft;

            pInputVal     -= ncolsRev;
            pWiVal        -= ncolsWiRev;
            pStateVal     -= ncolsRev;
            pGradWiVal    -= ncolsWiRev;
            pGradInputVal -= ncolsRev;
            pInGradHVal   -= ncolsRev;            
        } 
        *(pGradBias + col) = gbF;
        *(pGradBias + col + ncols) = gbR;
        *(pGradInit + col) = cur;
    }

    // gradB    
    gradBias.template reduceAlongDimension<simdOps::Sum<T>>(gradB, {0}, false, true);    // [1 x 4K]    

    // gradWeights     
    x->permutei({0, 2, 1});                                             // [time x bS x 2K] -> [time x 2K x bS]
    *gradWeights = mmul(*x, gradWi);                                    // [time x 2K x bS ] * [time x bS x 6K] = [time x 2K x 6K]

    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(sru_bi_bp) {

    auto inShape = inputShape->at(0);   // [time x bS x 2K]
    auto time    = inShape[1];
    auto bS   = inShape[2];
    auto inSize    = inShape[3] / 2;
    char order = shape::order(inShape);

    Nd4jLong *newShapeInfo1(nullptr), *newShapeInfo2(nullptr), *newShapeInfo3(nullptr), *newShapeInfo4(nullptr);
    ALLOCATE(newShapeInfo1, block.getWorkspace(), 10, Nd4jLong);
    ALLOCATE(newShapeInfo2, block.getWorkspace(), 10, Nd4jLong);
    ALLOCATE(newShapeInfo3, block.getWorkspace(), 8, Nd4jLong);
    ALLOCATE(newShapeInfo4, block.getWorkspace(), 8, Nd4jLong);    
    
    // gradInput
    newShapeInfo1[0] = 3;
    newShapeInfo1[1] = time;
    newShapeInfo1[2] = bS;
    newShapeInfo1[3] = 2*inSize;
    shape::updateStrides(newShapeInfo1, order);
    // gradWeights
    newShapeInfo2[0] = 3;        
    newShapeInfo2[1] = time;
    newShapeInfo2[2] = 2*inSize;
    newShapeInfo2[3] = 6*inSize;
    shape::updateStrides(newShapeInfo2, order);
    // gradB
    newShapeInfo3[0] = 2;
    newShapeInfo3[1] = 1;
    newShapeInfo3[2] = 4*inSize;
    shape::updateStrides(newShapeInfo3, order);
    // gradInit
    newShapeInfo4[0] = 2;        
    newShapeInfo4[1] = bS;
    newShapeInfo4[2] = 2*inSize;
    shape::updateStrides(newShapeInfo4, order);
    
    return SHAPELIST(newShapeInfo1, newShapeInfo2, newShapeInfo3, newShapeInfo4);
}   

}
}

#endif