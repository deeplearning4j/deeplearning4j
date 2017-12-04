//
// implementation of operations for Simple Recurrent Unit: arXiv:1709.02755v2 [cs.CL] 12 Sep 2017
//
// @author yurii@gmail.com
//

#include <op_boilerplate.h>
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>


namespace nd4j {
    namespace ops {

// return 2d array evaluated though last dimension interval t1-t2
template <typename T>
NDArray<T>* timestep(const NDArray<T>* const arr, const int t1, const int t2) {

        IndicesList list({ NDIndex::all(), NDIndex::all(), NDIndex::interval(t1,t2)});
        NDArray<T>* result = arr->subarray(list);     
        result->reshapei(result->ordering(), {arr->shapeOf()[0], arr->shapeOf()[1]} );

        return result;
}

template <typename T>
NDArray<T> sigmoid(const NDArray<T>& arr) {
    NDArray<T> result(arr.getShapeInfo(), arr.getWorkspace());
    (const_cast<NDArray<T>&>(arr)).template applyTransform<simdOps::Sigmoid<T>>(&result);

    return result;
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sru, 5, 2, false, 0, 0) {

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
    
    // multiplication matrix = matmul(weights,input)
    NDArray<T>* wi = NDArrayFactory<T>::mmulHelper(weights, input, nullptr, (T)1., (T)0.);      //       U [bS x 3K x N]    
    // wi.printShapeInfo();
    NDArray<T>* wiZ = wi->subarray( { NDIndex::all(), NDIndex::interval(0,K),     NDIndex::all() } );       // [bS x K x N]
    NDArray<T>* wiF = wi->subarray( { NDIndex::all(), NDIndex::interval(K,2*K),   NDIndex::all() } );       // forget gate [bS x K x N]
    NDArray<T>* wiR = wi->subarray( { NDIndex::all(), NDIndex::interval(2*K,3*K), NDIndex::all() } );       // reset gate [bS x K x N]
    NDArray<T>* bF  = bias->subarray( { NDIndex::all(), NDIndex::interval(0,K)  } );                        // biases for forget gate [1 x K]
    NDArray<T>* bR  = bias->subarray( { NDIndex::all(), NDIndex::interval(K,2*K)} );                        // biases for reset gate [1 x K]

    NDArray<T>* xt(nullptr), *zt(nullptr), *ft(nullptr), *rt(nullptr), *ct(nullptr), *ht(nullptr);
    NDArray<T>* ct_1 = init->dup(init->ordering());
    NDArray<T>* gct  = new NDArray<T>(state->ordering(), {bS, K});
    NDArray<T>* xmt  = input->dup(input->ordering());          
    //  input = input * mask
    if(applyMask)
        xmt->template applyBroadcast<simdOps::Multiply<T>>({0, 1}, mask, xmt, nullptr);            // apply mask    
        
    for (int t = 0; t < N; ++t) {
        xt = timestep(xmt, t, t+1);         // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        zt = timestep(wiZ, t, t+1);         // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        ft = timestep(wiF, t, t+1);         // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        rt = timestep(wiR, t, t+1);         // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        ct = timestep(state, t, t+1);       // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        ht = timestep(output, t, t+1);      // [bS x K x N] -> [bS x K x 1] -> [bS x K]
       
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

DECLARE_SHAPE_FN(sru) {

    int* inShape = inputShape->at(0);   // [bS x K x N]
    int rank = inShape[0];              // = 3
    int size = rank*2 + 4;
    int bS   = inShape[1];
    int K    = inShape[2];
    int N    = inShape[3];
    char order = (char)(inShape[size-1]);

    int* newShapeInfo1 = nullptr;
    int* newShapeInfo2 = nullptr;
    ALLOCATE(newShapeInfo1, block.getWorkspace(), size, int);
    ALLOCATE(newShapeInfo2, block.getWorkspace(), size, int);
    
    newShapeInfo1[0] = rank;        
    newShapeInfo1[1] = bS;
    newShapeInfo1[2] = K;
    newShapeInfo1[3] = N;
    
    shape::updateStrides(newShapeInfo1, order);
    memcpy(newShapeInfo2, newShapeInfo1, shape::shapeInfoByteLength(newShapeInfo1));
    
    return new ShapeList({newShapeInfo1, newShapeInfo2});
}   

//////////////////////////////////////////////////////////////////////////
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

        ft = sigmoid(ft + bF);
        rt = sigmoid(rt + bR);
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

    int* inShape = inputShape->at(0);   // [bS x K x N]
    int rank = inShape[0];              // = 3
    int size = rank*2 + 4;
    int bS   = inShape[1];
    int K    = inShape[2];
    int N    = inShape[3];
    char order = (char)(inShape[size-1]);

    int* newShapeInfo1 = nullptr;
    int* newShapeInfo2 = nullptr;
    ALLOCATE(newShapeInfo1, block.getWorkspace(), size, int);
    ALLOCATE(newShapeInfo2, block.getWorkspace(), size, int);
    
    newShapeInfo1[0] = rank;        
    newShapeInfo1[1] = bS;
    newShapeInfo1[2] = K;
    newShapeInfo1[3] = N;
    
    shape::updateStrides(newShapeInfo1, order);
    memcpy(newShapeInfo2, newShapeInfo1, shape::shapeInfoByteLength(newShapeInfo1));
    
    return new ShapeList({newShapeInfo1, newShapeInfo2});
}   

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sru_bp, 8, 4, true, 0, 0) {
    
    NDArray<T>* input    = INPUT_VARIABLE(0);                // X, input 3d tensor [bS x K x N], N - number of time steps, bS - batch size, K - number of features
    NDArray<T>* weights  = INPUT_VARIABLE(1);                // W, 2d tensor of weights [3K x K]
    NDArray<T>* bias     = INPUT_VARIABLE(2);                // B, row of biases with twice length [1 × 2*K]
    NDArray<T>* init     = INPUT_VARIABLE(3);                // C_{0}, 2d tensor of initial state [bS x K] at time t=0    
    NDArray<T>* state    = INPUT_VARIABLE(4);                // C, [bS x K x N]
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

    const int bS      = input->shapeOf()[0];                     
    const int K       = input->shapeOf()[1];                     
    const int N       = input->shapeOf()[2];                     // N - number of time steps
    
    NDArray<T>* gradBias = new NDArray<T>(input->ordering(), {bS, 2*K, N});
    NDArray<T>* gradU    = new NDArray<T>(input->ordering(), {bS, 3*K, N});
    NDArray<T>* gradHX   = new NDArray<T>(input->ordering(), {bS, K, N});
    NDArray<T>* gct      = new NDArray<T>(state->ordering(), {bS, K});    
    NDArray<T>* gradTanh = new NDArray<T>(state->ordering(), {bS, K});
    NDArray<T>* gradCt   = new NDArray<T>(state->ordering(), {bS, K});
    NDArray<T>* ftMinus  = new NDArray<T>(state->ordering(), {bS, K});
    NDArray<T>* rtMinus  = new NDArray<T>(state->ordering(), {bS, K});
    NDArray<T>* temp1    = new NDArray<T>(state->ordering(), {bS, K});    
    NDArray<T>* temp2    = new NDArray<T>(state->ordering(), {bS, K});       

    //  input = input * mask
    if(applyMask)
        input->template applyBroadcast<simdOps::Multiply<T>>({0, 1}, mask, input, nullptr);            // apply mask    
    // multiplication matrix wi = matmul(weights,input), U = WX
    NDArray<T>* wi = NDArrayFactory<T>::mmulHelper(weights, input, nullptr, (T)1., (T)0.);      // U [bS x 3K x N]    

    NDArray<T>* wiZ = wi->subarray( { NDIndex::all(), NDIndex::interval(0,K),     NDIndex::all() } );       // [bS x K x N]
    NDArray<T>* wiF = wi->subarray( { NDIndex::all(), NDIndex::interval(K,2*K),   NDIndex::all() } );       // forget gate [bS x K x N]
    NDArray<T>* wiR = wi->subarray( { NDIndex::all(), NDIndex::interval(2*K,3*K), NDIndex::all() } );       // reset gate [bS x K x N]
    NDArray<T>* bF  = bias->subarray( { NDIndex::all(), NDIndex::interval(0,K)  } );                        // biases for forget gate [1 x K]
    NDArray<T>* bR  = bias->subarray( { NDIndex::all(), NDIndex::interval(K,2*K)} );                        // biases for reset gate [1 x K]
    NDArray<T>* gradBF = gradBias->subarray( { NDIndex::all(), NDIndex::interval(0,K),   NDIndex::all() } );   // [bS x K x N]
    NDArray<T>* gradBR = gradBias->subarray( { NDIndex::all(), NDIndex::interval(K,2*K), NDIndex::all() } );   // [bS x K x N]
    NDArray<T>* gradUZ = gradU->subarray( { NDIndex::all(), NDIndex::interval(0,K),     NDIndex::all() } ); // [bS x K x N]
    NDArray<T>* gradUF = gradU->subarray( { NDIndex::all(), NDIndex::interval(K,2*K),   NDIndex::all() } ); // [bS x K x N]
    NDArray<T>* gradUR = gradU->subarray( { NDIndex::all(), NDIndex::interval(2*K,3*K), NDIndex::all() } ); // [bS x K x N]


    NDArray<T>* xt(nullptr), *zt(nullptr), *ft(nullptr), *rt(nullptr), *ct(nullptr), *inGradHt(nullptr), *gradBFt(nullptr), 
                *gradBRt(nullptr), *ct_1(nullptr), *gradHXt(nullptr), *gradURt(nullptr), *gradUFt(nullptr), *gradUZt(nullptr);

    for (int t = N-1; t >=0 ; --t) {           
        // initialization
        xt = timestep(input, t, t+1);               // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        zt = timestep(wiZ, t, t+1);                 // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        ft = timestep(wiF, t, t+1);                 // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        rt = timestep(wiR, t, t+1);                 // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        ct = timestep(state, t, t+1);               // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        inGradHt = timestep(inGradH, t, t+1);       // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        gradBRt  = timestep(gradBR, t, t+1);        // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        gradBFt  = timestep(gradBF, t, t+1);        // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        gradHXt  = timestep(gradHX, t, t+1);        // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        gradUZt  = timestep(gradUZ, t, t+1);        // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        gradUFt  = timestep(gradUF, t, t+1);        // [bS x K x N] -> [bS x K x 1] -> [bS x K]
        gradURt  = timestep(gradUR, t, t+1);        // [bS x K x N] -> [bS x K x 1] -> [bS x K]                        

        if(t != 0)
            ct_1  = timestep(state, t-1, t);        // previous c_{t-1} 
        else
            ct_1 = init->dup(init->ordering());
        
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
    NDArray<T>* weightsT = weights->transpose();                                            // [K x 3K]
    NDArrayFactory<T>::mmulHelper(weightsT, gradU, gradX, (T)1., (T)0.);                    // [bS x K x N]    
    gradX->template applyPairwiseTransform<simdOps::Add<T>>(gradHX, gradX, nullptr);        // + grad_highway_x
    if(applyMask)
        gradX->template applyBroadcast<simdOps::Multiply<T>>({0,1}, mask, gradX, nullptr);  // apply mask

    // gradB    
    NDArray<T>* temp3 = gradBias->template reduceAlongDimension<simdOps::Sum<T>>({0,2});    // [1 x 2K]
    gradB->assign(temp3);

    // gradW [bS x 3K x K]
    input->permutei({0, 2, 1});                                               // [bS x N x K]
    NDArrayFactory<T>::mmulHelper(gradU, input, gradW, (T)1., (T)0.);          // [bS x 3K x K]

    delete gradUR; delete gradBF; delete gradUZ; delete gradUF; delete gradBR;

    delete gct;   delete gradU; delete gradHX; delete wiZ; delete wiF; delete wiR; delete bF; delete bR;
    delete temp1; delete temp2; delete temp3; delete gradCt; delete wi;
    delete gradTanh; delete ftMinus; delete rtMinus; delete weightsT; delete gradBias;
    
    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(sru_bp) {

    int* inShape = inputShape->at(0);   // [bS x K x N]
    int bS   = inShape[1];
    int K    = inShape[2];
    int N    = inShape[3];
    char order = (char)(inShape[9]);

    int *newShapeInfo1(nullptr), *newShapeInfo2(nullptr), *newShapeInfo3(nullptr), *newShapeInfo4(nullptr);
    ALLOCATE(newShapeInfo1, block.getWorkspace(), 10, int);
    ALLOCATE(newShapeInfo2, block.getWorkspace(), 10, int);
    ALLOCATE(newShapeInfo3, block.getWorkspace(), 8, int);
    ALLOCATE(newShapeInfo4, block.getWorkspace(), 8,  int);    
    
    newShapeInfo1[0] = 3;
    newShapeInfo1[1] = bS;
    newShapeInfo1[2] = K;
    newShapeInfo1[3] = N;
    shape::updateStrides(newShapeInfo1, order);

    newShapeInfo2[0] = 3;        
    newShapeInfo2[1] = bS;
    newShapeInfo2[2] = 3*K;
    newShapeInfo2[3] = K;
    shape::updateStrides(newShapeInfo2, order);

    newShapeInfo3[0] = 2;
    newShapeInfo3[1] = 1;
    newShapeInfo3[2] = 2*K;    
    shape::updateStrides(newShapeInfo3, order);

    newShapeInfo4[0] = 2;        
    newShapeInfo4[1] = bS;
    newShapeInfo4[2] = K;    
    shape::updateStrides(newShapeInfo4, order);
    
    return new ShapeList({newShapeInfo1, newShapeInfo2, newShapeInfo3, newShapeInfo4});
}   
 

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sru_bp_logic, 8, 4, true, 0, 0) {
    
    NDArray<T>* input    = INPUT_VARIABLE(0);                // X, input 3d tensor [bS x K x N], N - number of time steps, bS - batch size, K - number of features
    NDArray<T>* weights  = INPUT_VARIABLE(1);                // W, 2d tensor of weights [3K x K]
    NDArray<T>* bias     = INPUT_VARIABLE(2);                // B, row of biases with twice length [1 × 2*K]
    NDArray<T>* init     = INPUT_VARIABLE(3);                // C_{0}, 2d tensor of initial state [bS x K] at time t=0    
    NDArray<T>* state    = INPUT_VARIABLE(4);                // C, [bS x K x N]
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

    const int bS      = input->shapeOf()[0];                     
    const int K       = input->shapeOf()[1];                     
    const int N       = input->shapeOf()[2];                     // N - number of time steps
    
    const NDArray<T> bF = (*bias)({ {}, {0,  K} });                                 // biases for forget gate [1 x K]
    const NDArray<T> bR = (*bias)({ {}, {K,2*K} });                                 // biases for reset  gate [1 x K]    
    NDArray<T> gradBias(input->ordering(),   {bS, 2*K, N}, block.getWorkspace());
    NDArray<T> gradU   (input->ordering(),   {bS, 3*K, N}, block.getWorkspace());
    NDArray<T> gradHX  (input->ordering(),   {bS,   K, N}, block.getWorkspace());
    NDArray<T> gct     (state->ordering(),   {bS, K},      block.getWorkspace());

    NDArray<T> gradBFt(block.getWorkspace());
    NDArray<T> gradBRt(block.getWorkspace());
    NDArray<T> gradUZt(block.getWorkspace()); 
    NDArray<T> xt(block.getWorkspace());
    NDArray<T> zt(block.getWorkspace()); 
    NDArray<T> ft(block.getWorkspace()); 
    NDArray<T> rt(block.getWorkspace());     
    NDArray<T> ct(block.getWorkspace());
    NDArray<T> ct_1(block.getWorkspace());
    NDArray<T> gradHXt(block.getWorkspace());
    NDArray<T> inGradHt(block.getWorkspace());
    NDArray<T> gradTanh(block.getWorkspace());
    NDArray<T> gradCt(block.getWorkspace());
    NDArray<T> ftMinus(block.getWorkspace());
    NDArray<T> rtMinus(block.getWorkspace());

    //  input = input * mask    
    if(applyMask)
        input->template applyBroadcast<simdOps::Multiply<T>>({0, 1}, mask, input, nullptr);             // apply mask    
    // multiplication matrix wi = matmul(weights,input), U = WX
    const NDArray<T> wi = mmul(*weights, *input);                                                   //  U [bS x 3K x N]            

    for (int t = N-1; t >=0 ; --t) {           
        // initialization
        xt =         (*input)({ {}, {},        {t,t+1} }); xt.reshapei(xt.ordering(), {bS, K});           // [bS x K  x N] -> [bS x K x 1] -> [bS x K]
        zt =               wi({ {}, {0,    K}, {t,t+1} }); zt.reshapei(zt.ordering(), {bS, K});           // [bS x 3K x N] -> [bS x K x 1] -> [bS x K]
        ft =               wi({ {}, {K,  2*K}, {t,t+1} }); ft.reshapei(ft.ordering(), {bS, K});           // [bS x 3K x N] -> [bS x K x 1] -> [bS x K]
        rt =               wi({ {}, {2*K,3*K}, {t,t+1} }); rt.reshapei(rt.ordering(), {bS, K});           // [bS x 3K x N] -> [bS x K x 1] -> [bS x K]
        ct =         (*state)({ {}, {},        {t,t+1} }); ct.reshapei(ct.ordering(), {bS, K});           // [bS x K  x N] -> [bS x K x 1] -> [bS x K]        
        inGradHt = (*inGradH)({ {}, {},        {t,t+1} }); inGradHt.reshapei(xt.ordering(), {bS, K});     // [bS x K  x N] -> [bS x K x 1] -> [bS x K]
        if(t != 0)
            { ct_1 = (*state)({ {}, {}, {t-1,t} }); ct_1.reshapei(ct_1.ordering(), {bS, K}); }            // previous c_{t-1} 
        else
            ct_1 = *init;                   
        
        ///////////////// forward
        // ft = sigmoid(ft + bf), rt = sigmoid(rt + bR)
        ft = sigmoid(ft + bF);
        rt = sigmoid(rt + bR);        
        // TODO T val = (activation_type == 1) ? tanh(cur) : ((activation_type == 2) ? reluf(cur) : cur );
        ct.template applyTransform<simdOps::Tanh<T>>(&gct);        

        ///////////////// backward
        // bR, *grad_brt_ptr = inGradHt * (g_ct - xt) * (1.0f - rt) * rt;
        // ftMinus = -ft + (T)1.;
        ftMinus = (T)1. - ft;
        rtMinus = (T)1. - rt;
        gradBRt = inGradHt * (gct - xt) * rtMinus * rt;
        // bF, TODO - tanh            
        gradTanh = (T)1. - gct * gct;
        gradCt = inGradHt * rt * gradTanh;
        gradBFt = (gradCt + *inGradCt) * (ct_1 - zt) * ftMinus * ft;        
        // x_t (highway connection), gradHXt = inGradHt * (1.0f - rt);
        gradHXt = inGradHt * rtMinus;

        // U_t, gradUZt = (inGradHt * rt * grad_tanh + inGradCt) * (1.0f - ft);
        gradUZt = (inGradHt * rt * gradTanh + *inGradCt) * ftMinus;

        // c_{t-1}, inGradCt = (gradCt + inGradCt) * ft;
        *inGradCt = (gradCt + *inGradCt) * ft;
        
        // save results        
        gradBias.assign(gradBFt, {{}, {0,     K}, {t,t+1}} );
        gradBias.assign(gradBRt, {{}, {K,   2*K}, {t,t+1}} );
        gradU.assign(gradUZt,    {{}, {0,     K}, {t,t+1}} );
        gradU.assign(gradBFt,    {{}, {K,   2*K}, {t,t+1}} );
        gradU.assign(gradBRt,    {{}, {2*K, 3*K}, {t,t+1}} );       
        gradHX.assign(gradHXt,   {{}, {        }, {t,t+1}} );              
    }

    // gradInit
    gradInit->assign(inGradCt);
    // gradX 
    weights->transposei();                                                               // [K x 3K]
    *gradX = mmul(*weights, gradU) + gradHX;        
    if(applyMask)
        gradX->template applyBroadcast<simdOps::Multiply<T>>({0,1}, mask, gradX, nullptr);       // apply mask

    // gradB    
    gradBias.template reduceAlongDimension<simdOps::Sum<T>>(gradB, {0,2});    // [1 x 2K]    

    // gradW [bS x 3K x K]
    input->permutei({0, 2, 1});                                               // [bS x N x K]
    *gradW = mmul(gradU, *input);    
    
    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(sru_bp_logic) {

    int* inShape = inputShape->at(0);   // [bS x K x N]
    int bS   = inShape[1];
    int K    = inShape[2];
    int N    = inShape[3];
    char order = (char)(inShape[9]);

    int *newShapeInfo1(nullptr), *newShapeInfo2(nullptr), *newShapeInfo3(nullptr), *newShapeInfo4(nullptr);
    ALLOCATE(newShapeInfo1, block.getWorkspace(), 10, int);
    ALLOCATE(newShapeInfo2, block.getWorkspace(), 10, int);
    ALLOCATE(newShapeInfo3, block.getWorkspace(), 8, int);
    ALLOCATE(newShapeInfo4, block.getWorkspace(), 8,  int);    
    
    newShapeInfo1[0] = 3;
    newShapeInfo1[1] = bS;
    newShapeInfo1[2] = K;
    newShapeInfo1[3] = N;
    shape::updateStrides(newShapeInfo1, order);

    newShapeInfo2[0] = 3;        
    newShapeInfo2[1] = bS;
    newShapeInfo2[2] = 3*K;
    newShapeInfo2[3] = K;
    shape::updateStrides(newShapeInfo2, order);

    newShapeInfo3[0] = 2;
    newShapeInfo3[1] = 1;
    newShapeInfo3[2] = 2*K;    
    shape::updateStrides(newShapeInfo3, order);

    newShapeInfo4[0] = 2;        
    newShapeInfo4[1] = bS;
    newShapeInfo4[2] = K;    
    shape::updateStrides(newShapeInfo4, order);
    
    return new ShapeList({newShapeInfo1, newShapeInfo2, newShapeInfo3, newShapeInfo4});
}   

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sru_bi, 5, 2, true, 0, 0) {

    NDArray<T>* input   = INPUT_VARIABLE(0);                // X, input 3d tensor [N x bS x 2K], N - number of time steps, bS - batch size, K - number of features
    NDArray<T>* weights = INPUT_VARIABLE(1);                // W, 2d tensor of weights [2K x 6K]
    NDArray<T>* bias    = INPUT_VARIABLE(2);                // B, row of biases with twice length [1 × 4K]
    NDArray<T>* init    = INPUT_VARIABLE(3);                // C_{0}, 2d tensor of initial state [bS x 2K] at time t=0
    NDArray<T>* mask    = nullptr;                          // optional, 2d tensor of dropout mask [bS x 2K]

    bool applyMask = false;        
    if (block.width() > 4) {
        mask = INPUT_VARIABLE(4);   
        applyMask = true;
    }

    NDArray<T>* output = OUTPUT_VARIABLE(0);                // h_t, [N x bS x 2K]
    NDArray<T>* state  = OUTPUT_VARIABLE(1);                // c_t, [N x bS x 2K]
    
    const int N      = input->shapeOf()[0];                     // N - number of time steps
    const int bS     = input->shapeOf()[1];                     // bS - batch size
    const int K      = input->shapeOf()[2] / 2;                 // K - number of features
  
    //  input = input * mask    
    if(applyMask)
        input->template applyBroadcast<simdOps::Multiply<T>>({1, 2}, mask, input, nullptr);             // apply mask    
    // U = input * weights 
    NDArray<T> wi = mmul(*input, *weights);                    //  U [N x bS x 6K]                

    const int d2      = 2*K;
    const int ncols   = bS*d2;     
    const int ncolsWi = 3*ncols;    

    T* const pInput  = input->getBuffer();
    T* const pWi     = wi.getBuffer();
    T* const pBias   = bias->getBuffer();
    T* const pInit   = init->getBuffer();
    T* const pMask   = mask->getBuffer();
    T* const pOutput = output->getBuffer();
    T* const pState  = state->getBuffer();

    int ncolsRev, ncolsWiRev;                   // for reverse direction
    T maskVal, cur, bF, bR, ft, rt, val;
    T *pInputVal(nullptr), *pWiVal(nullptr), *pOutputVal(nullptr), *pStateVal(nullptr);
    bool flip = false;

    for (int col = 0; col < ncols; ++col) {           
        
        flip       = (col%d2) >= K;
        maskVal    = applyMask ? *(pMask + col) : (T)1.;
        cur        = *(pInit + col);
        bF         = *(pBias + col%d2);
        bR         = *(pBias + col%d2 + d2);
        pWiVal     = pWi     + 3*col;
        pInputVal  = pInput  + col;
        pOutputVal = pOutput + col;
        pStateVal  = pState  + col;

        if (flip) {
            pInputVal  += (N-1)*ncols;
            pWiVal     += (N-1)*ncolsWi;
            pOutputVal += (N-1)*ncols;
            pStateVal  += (N-1)*ncols;
        }

        ncolsRev   = flip ? -ncols   : ncols;
        ncolsWiRev = flip ? -ncolsWi : ncolsWi;
        
        for (int t = 0; t < N; ++t) {
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

    int* inShape = inputShape->at(0);   // [N x bS x 2K ]
    int rank = inShape[0];              // = 3
    int size = rank*2 + 4;        
    int N    = inShape[1];
    int bS   = inShape[2];
    int K    = inShape[3] / 2;
    
    char order = (char)(inShape[size-1]);

    int* newShapeInfo1 = nullptr;
    int* newShapeInfo2 = nullptr;
    ALLOCATE(newShapeInfo1, block.getWorkspace(), size, int);
    ALLOCATE(newShapeInfo2, block.getWorkspace(), size, int);
    
    newShapeInfo1[0] = rank;        
    newShapeInfo1[1] = N;
    newShapeInfo1[2] = bS;
    newShapeInfo1[3] = 2*K;
    
    shape::updateStrides(newShapeInfo1, order);
    memcpy(newShapeInfo2, newShapeInfo1, shape::shapeInfoByteLength(newShapeInfo1));
    
    return new ShapeList({newShapeInfo1, newShapeInfo2});
}   


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sru_bi_bp, 8, 4, true, 0, 0) {
    
    NDArray<T>* input    = INPUT_VARIABLE(0);                // X, input 3d tensor [N x bS x 2K], N - number of time steps, bS - batch size, K - number of features
    NDArray<T>* weights  = INPUT_VARIABLE(1);                // W, 2d tensor of weights [2K x 6K]
    NDArray<T>* bias     = INPUT_VARIABLE(2);                // B, row of biases with twice length [1 × 4K]
    NDArray<T>* init     = INPUT_VARIABLE(3);                // C_{0}, 2d tensor of initial state [bS x 2K] at time t=0
    NDArray<T>* state    = INPUT_VARIABLE(4);                // C, [N x bS x 2K]
    NDArray<T>* inGradCt = INPUT_VARIABLE(5);                // [bS x 2K]
    NDArray<T>* inGradH  = INPUT_VARIABLE(6);                // [N x bS x 2K]
    NDArray<T>* mask     = nullptr;                          // optional,  2d tensor of dropout mask [bS x 2K]

    bool applyMask = false;        
    if (block.width() > 7) {
        mask = INPUT_VARIABLE(7);   
        applyMask = true;
    }

    NDArray<T>* gradInput   = OUTPUT_VARIABLE(0);              // [N x bS x 2K]
    NDArray<T>* gradWeights = OUTPUT_VARIABLE(1);              // [N x 2K x 6K]
    NDArray<T>* gradB       = OUTPUT_VARIABLE(2);              // [1 x 4K]
    NDArray<T>* gradInit    = OUTPUT_VARIABLE(3);              // [bS x 2K]

    const int N       = input->shapeOf()[0];                     // N - number of time steps
    const int bS      = input->shapeOf()[1];                     
    const int K       = input->shapeOf()[2] / 2;                     

    //  input = input * mask    
    if(applyMask)
        input->template applyBroadcast<simdOps::Multiply<T>>({1, 2}, mask, input, nullptr);             // apply mask    
    // U = input * weights 
    NDArray<T> wi = mmul(*input, *weights);                    //  [N x bS x 2K] * [2K x 6K] = [N x bS x 6K]                

    NDArray<T> gradBias(input->ordering(), {bS, 4*K},    block.getWorkspace());
    NDArray<T> gradWi  (input->ordering(), {N, bS, 6*K}, block.getWorkspace());
    
    const int d2      = 2*K;
    const int ncols   = bS*d2;     
    const int ncolsWi = 3*ncols;    

    T* const pInput     = input->getBuffer();
    T* const pWi        = wi.getBuffer();
    T* const pBias      = bias->getBuffer();
    T* const pInit      = init->getBuffer();
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

        flip          = (col%d2) >= K;
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
            pInputVal     += (N-1)*ncols;
            pWiVal        += (N-1)*ncolsWi;
            pStateVal     += (N-1)*ncols;
            pInGradHVal   += (N-1)*ncols;
            pGradWiVal    += (N-1)*ncolsWi;
            pGradInputVal += (N-1)*ncols;
        }

        ncolsRev   = flip ? -ncols   : ncols;
        ncolsWiRev = flip ? -ncolsWi : ncolsWi;
        
        for (int t = 0; t < N; ++t) {
            // evaluate sigmoids 
            ft = ((T)1.)/((T)1. + nd4j::math::nd4j_exp<T>(-(*(pWiVal + 1) + bF)));
            rt = ((T)1.)/((T)1. + nd4j::math::nd4j_exp<T>(-(*(pWiVal + 2) + bR)));
            
            val     = nd4j::math::nd4j_tanh<T>(*pStateVal);            
            prevVal = (t < N-1) ? (*(pStateVal - ncolsRev)) : (*(pInit + col));
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
    gradBias.template reduceAlongDimension<simdOps::Sum<T>>(gradB, {0});    // [1 x 4K]    

    // gradWeights     
    input->permutei({0, 2, 1});                                             // [N x bS x 2K] -> [N x 2K x bS]    
    *gradWeights = mmul(*input, gradWi);                                    // [N x 2K x bS ] * [N x bS x 6K] = [N x 2K x 6K]    

    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(sru_bi_bp) {

    int* inShape = inputShape->at(0);   // [N x bS x 2K]
    int N    = inShape[1];
    int bS   = inShape[2];
    int K    = inShape[3] / 2;    
    char order = (char)(inShape[9]);

    int *newShapeInfo1(nullptr), *newShapeInfo2(nullptr), *newShapeInfo3(nullptr), *newShapeInfo4(nullptr);
    ALLOCATE(newShapeInfo1, block.getWorkspace(), 10, int);
    ALLOCATE(newShapeInfo2, block.getWorkspace(), 10, int);
    ALLOCATE(newShapeInfo3, block.getWorkspace(), 8, int);
    ALLOCATE(newShapeInfo4, block.getWorkspace(), 8, int);    
    
    // gradInput
    newShapeInfo1[0] = 3;
    newShapeInfo1[1] = N;
    newShapeInfo1[2] = bS;
    newShapeInfo1[3] = 2*K;
    shape::updateStrides(newShapeInfo1, order);
    // gradWeights
    newShapeInfo2[0] = 3;        
    newShapeInfo2[1] = N;
    newShapeInfo2[2] = 2*K;
    newShapeInfo2[3] = 6*K;
    shape::updateStrides(newShapeInfo2, order);
    // gradB
    newShapeInfo3[0] = 2;
    newShapeInfo3[1] = 1;
    newShapeInfo3[2] = 4*K;    
    shape::updateStrides(newShapeInfo3, order);
    // gradInit
    newShapeInfo4[0] = 2;        
    newShapeInfo4[1] = bS;
    newShapeInfo4[2] = 2*K;
    shape::updateStrides(newShapeInfo4, order);
    
    return new ShapeList({newShapeInfo1, newShapeInfo2, newShapeInfo3, newShapeInfo4});
}   

}
}
