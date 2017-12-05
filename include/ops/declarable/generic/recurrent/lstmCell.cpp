//
// implementation of operation for LSTM cell with peep hole connections:
// http://www.bioinf.jku.at/publications/older/2604.pdf
// S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
// and 
// https://research.google.com/pubs/archive/43905.pdf
// Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory recurrent neural network architectures for large scale acoustic modeling." INTERSPEECH, 2014.
//
// created by Yurii Shyrma on 30.11.2017
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {


//////////////////////////////////////////////////////////////////////////
template <typename T>
static NDArray<T> sigmoid(const NDArray<T>& arr) {    
    
    return (const_cast<NDArray<T>&>(arr)).template transform<simdOps::Sigmoid<T>>();    
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static NDArray<T> activation(const NDArray<T>& arr) {    
    
    return (const_cast<NDArray<T>&>(arr)).template transform<simdOps::Tanh<T>>();    
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void clipping(NDArray<T>* arr, T limit) {    
    
    if(limit < (T)0.)
        limit *= (T)(-1.);

    auto clip = LAMBDA_T(value, limit) {
        if(value < -limit || value > limit)
            value = limit;
        return value; 
    };

    arr->applyLambda(clip);    
}


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(lstmCell, 8, 2, false, 3, 2) {

    NDArray<T>* xt   = INPUT_VARIABLE(0);                   // input [batchSize x inSize]
    NDArray<T>* ht_1 = INPUT_VARIABLE(1);                   // previous cell output [batchSize x numProj],  that is at previous time step t-1, in case of projection=false -> numProj=numUnits!!! 
    NDArray<T>* ct_1 = INPUT_VARIABLE(2);                   // previous cell state  [batchSize x numUnits], that is at previous time step t-1   

    NDArray<T>* Wx   = INPUT_VARIABLE(3);                   // input-to-hidden  weights, [inSize  x 4*numUnits] 
    NDArray<T>* Wh   = INPUT_VARIABLE(4);                   // hidden-to-hidden weights, [numProj x 4*numUnits] 
    NDArray<T>* Wc   = INPUT_VARIABLE(5);                   // diagonal weights for peephole connections [1 x 3*numUnits] 
    NDArray<T>* Wp   = INPUT_VARIABLE(6);                   // projection weights [numUnits x numProj] 
    NDArray<T>* b    = INPUT_VARIABLE(7);                   // biases, [1 x 4*numUnits] 
    
    NDArray<T>* ht   =  OUTPUT_VARIABLE(0);                  // current cell output [batchSize x numProj], that is at current time step t
    NDArray<T>* ct   =  OUTPUT_VARIABLE(1);                  // current cell state  [batchSize x numUnits], that is at current time step t
    
    const bool peephole   = (bool)INT_ARG(0);               // if true, provide peephole connections
    const bool projection = (bool)INT_ARG(1);               // if true, then projection is performed, if false then numProj==numUnits is mandatory!!!!
    T clippingCellValue   = T_ARG(0);                       // clipping value for ct, if it is not equal to zero, then cell state is clipped
    T clippingProjValue   = T_ARG(1);                       // clipping value for projected ht, if it is not equal to zero, then projected cell output is clipped
    const T forgetBias    = T_ARG(2);

    const int numUnits  = ct_1->sizeAt(1);
    
    NDArray<T> z = mmul(*xt, *Wx) + mmul(*ht_1, *Wh) + *b;       // [batchSize x 4*numUnits] + [batchSize x 4*numUnits] + [1 x 4*numUnits] = [batchSize x 4*numUnits]    
    
    NDArray<T> zit = z({{},{0,            numUnits}});      // z for input gate,  = mmul(Wxi,xt) + mmul(Whi,ht_1) + bi    = [batchSize x numUnits]
    NDArray<T> zft = z({{},{numUnits,   2*numUnits}});      // z for forget gate, = mmul(Wxf,xt) + mmul(Whf,ht_1) + bf    = [batchSize x numUnits]
    NDArray<T> zct = z({{},{2*numUnits, 3*numUnits}});      // z for cell state,  = mmul(Wxc,xt) + mmul(Whc,ht_1) + bc    = [batchSize x numUnits]     
    NDArray<T> zot = z({{},{3*numUnits, 4*numUnits}});      // z for output gate, = mmul(Wxo,xt) + mmul(Who,ht_1) + bo    = [batchSize x numUnits] 

    if(peephole) {                                              // add peephole connections: z  +  ct_1*Wc
        zit += (*ct_1) * (*Wc)({{},{0,          numUnits}});    // add peephole connections to input gate
        zft += (*ct_1) * (*Wc)({{},{numUnits, 2*numUnits}});    // add peephole connections to forget gate
    }

    // current sell state = ft*ct_1 + it*activation(mmul(Wxc,xt) + mmul(Whc,ht_1) + bc
    *ct = sigmoid<T>(zft + forgetBias) * (*ct_1) + sigmoid<T>(zit) * activation<T>(zct);
    
    // if clipping value is provided then cell state is clipped by this value prior to the cell output activation
    if(clippingCellValue != (T)0.)
        clipping(ct, clippingCellValue);

    if(peephole) 
        zot += (*ct) * (*Wc)({{},{2*numUnits, 3*numUnits}});            // add peephole connections to output gate zot + ct*Wc

    // current cell output = ot*activation(ct)   
    NDArray<T> htNoPeepHole = sigmoid<T>(zot) * activation<T>(*ct);      // = [batchSize x numUnits]

    // apply projection
    if(projection) {
        *ht = mmul(htNoPeepHole, *Wp);                                  // [batchSize x numUnits] * [ numUnits x numProj] = [batchSize x numProj]
        // if clipping projection is provided then projected cell output state is clipped by this value 
        if(clippingProjValue != (T)0.)
            clipping(ht, clippingProjValue);
    }
    else
        ht->assign(&htNoPeepHole);     

    

    return ND4J_STATUS_OK;
}



DECLARE_SHAPE_FN(lstmCell) {

    const int batchSize   = (INPUT_VARIABLE(0))->sizeAt(0);
    const int inSize      = (INPUT_VARIABLE(0))->sizeAt(1);
    const int numProj     = (INPUT_VARIABLE(1))->sizeAt(1);
    const int numUnits    = (INPUT_VARIABLE(2))->sizeAt(1);
    const bool projection = (bool)INT_ARG(1);

    // check shapes of previous cell output and previous cell state
    for(int i = 1; i <=2; ++i)
        if((INPUT_VARIABLE(i))->sizeAt(0) != batchSize)
            throw "CUSTOM_OP lstmCell: the shape[0] of previous cell output or previous cell state must be equal to batch size !";
    
    // check shape of input-to-hidden  weights
    if(!INPUT_VARIABLE(3)->isSameShape({inSize, 4*numUnits}))
        throw "CUSTOM_OP lstmCell: the shape of input-to-hidden weights is wrong !";

    // check shape of hidden-to-hidden  weights
    if(!INPUT_VARIABLE(4)->isSameShape({numProj, 4*numUnits}))
        throw "CUSTOM_OP lstmCell: the shape of hidden-to-hidden weights is wrong !";

    // check shape of diagonal weights
    if(!INPUT_VARIABLE(5)->isSameShape({1, 3*numUnits}))
        throw "CUSTOM_OP lstmCell: the shape of diagonal weights is wrong !";

    // check shape of projection weights
    if(!INPUT_VARIABLE(6)->isSameShape({numUnits, numProj}))
        throw "CUSTOM_OP lstmCell: the shape of projection weights is wrong !";

    // check shape of biases
    if(!INPUT_VARIABLE(7)->isSameShape({1, 4*numUnits}))
        throw "CUSTOM_OP lstmCell: the shape of biases is wrong !";

    if(!projection && numUnits != numProj)
        throw "CUSTOM_OP lstmCell: projection option is switched of, and in this case output dimensionality for the projection matrices (numProj) must be equal to number of units in lstmCell !";

    // evaluate output shapeInfos
    int *outShapeInfo1(nullptr), *outShapeInfo2(nullptr);
    ALLOCATE(outShapeInfo1, block.getWorkspace(), 8, int);
    ALLOCATE(outShapeInfo2, block.getWorkspace(), 8, int);
            
    outShapeInfo1[0] = outShapeInfo2[0] = 2;
    outShapeInfo1[1] = outShapeInfo2[1] = batchSize;
    outShapeInfo1[2] = numProj;
    outShapeInfo2[2] = numUnits;
    
    shape::updateStrides(outShapeInfo1, (INPUT_VARIABLE(1))->ordering());
    shape::updateStrides(outShapeInfo2, (INPUT_VARIABLE(2))->ordering());
         
    return new ShapeList({outShapeInfo1, outShapeInfo2});
}   








}
}

