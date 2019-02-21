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
// @author Yurii Shyrma, created on 14.02.2018
//

// implementation of operation for LSTM cell with peep hole connections:
// http://www.bioinf.jku.at/publications/older/2604.pdf
// S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
// and 
// https://research.google.com/pubs/archive/43905.pdf
// Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory recurrent neural network architectures for large scale acoustic modeling." INTERSPEECH, 2014.


#include<ops/declarable/helpers/lstm.h>
#include <VariableSpace.h>
#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>
#include <ops/declarable/helpers/legacy_helpers.h>
#include <array/NDArrayList.h>
#include <iterator>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
static FORCEINLINE NDArray sigmoid(const NDArray& arr) {
    
    return (const_cast<NDArray&>(arr)).transform(transform::Sigmoid);
}

//////////////////////////////////////////////////////////////////////////
static FORCEINLINE NDArray activation(const NDArray& arr) {
    
    return (const_cast<NDArray&>(arr)).transform(transform::Tanh);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void clipping(NDArray* arr, T limit) {
    
    if(limit < (T)0.f)
        limit *= (T)(-1.f);

    /*
    auto clip = LAMBDA_T(value, limit) {
        if(value < -limit || value > limit)
            value = limit;
        return value; 
    };

    arr->applyLambda(clip);
    */
    arr->applyScalar(scalar::LstmClip, limit);
}

//////////////////////////////////////////////////////////////////////////
void lstmCell(const NDArray* xt, const NDArray* ht_1, const NDArray* ct_1, const NDArray* Wx, const NDArray* Wh, const NDArray* Wc, const NDArray* Wp, const NDArray* b,
              NDArray* ht, NDArray* ct, const std::vector<double>& params) {

    // xt   input [bS x inSize]
    // ht_1 previous cell output [bS x numProj],  that is at previous time step t-1, in case of projection=false -> numProj=numUnits!!!
    // ct_1 previous cell state  [bS x numUnits], that is at previous time step t-1

    // Wx   input-to-hidden  weights, [inSize  x 4*numUnits]
    // Wh   hidden-to-hidden weights, [numProj x 4*numUnits]
    // Wc   diagonal weights for peephole connections [3*numUnits]
    // Wp   projection weights [numUnits x numProj]
    // b    biases, [4*numUnits]
    
    // ht  current cell output [bS x numProj], that is at current time step t
    // ct  current cell state  [bS x numUnits], that is at current time step t
    
    const bool peephole   = (bool)params[0];        // if true, provide peephole connections
    const bool projection = (bool)params[1];        // if true, then projection is performed, if false then numProj==numUnits is mandatory!!!!
    double clippingCellValue   = params[2];              // clipping value for ct, if it is not equal to zero, then cell state is clipped
    double clippingProjValue   = params[3];              // clipping value for projected ht, if it is not equal to zero, then projected cell output is clipped
    const double forgetBias    = params[4];

    const int bS   = xt->sizeAt(0);
    const int inSize      = xt->sizeAt(1);
    const int numProj     = ht_1->sizeAt(1);
    const int numUnits    = ct_1->sizeAt(1);    
    
    auto z = mmul(*xt, *Wx) + mmul(*ht_1, *Wh) + *b;      // [bS x 4*numUnits] + [bS x 4*numUnits] + [1 x 4*numUnits] = [bS x 4*numUnits]
    
    auto zit = z({0,0, 0,            numUnits});      	// z for input gate,  = mmul(Wxi,xt) + mmul(Whi,ht_1) + bi    = [bS x numUnits]
    auto zft = z({0,0, numUnits,   2*numUnits});      	// z for forget gate, = mmul(Wxf,xt) + mmul(Whf,ht_1) + bf    = [bS x numUnits]
    auto zct = z({0,0, 2*numUnits, 3*numUnits});      	// z for cell state,  = mmul(Wxc,xt) + mmul(Whc,ht_1) + bc    = [bS x numUnits]
    auto zot = z({0,0, 3*numUnits, 4*numUnits});      	// z for output gate, = mmul(Wxo,xt) + mmul(Who,ht_1) + bo    = [bS x numUnits]

    if(peephole) {                                              // add peephole connections: z  +  ct_1*Wc
        zit += (*ct_1) * (*Wc)({0,          numUnits});       // add peephole connections to input gate
        zft += (*ct_1) * (*Wc)({numUnits, 2*numUnits});       // add peephole connections to forget gate
    }

    // current sell state = ft*ct_1 + it*activation(mmul(Wxc,xt) + mmul(Whc,ht_1) + bc
    ct->assign( sigmoid(zft + forgetBias) * (*ct_1) + sigmoid(zit) * activation(zct) );
    
    // if clipping value is provided then cell state is clipped by this value prior to the cell output activation
    if(clippingCellValue != 0.)
        clipping(ct, clippingCellValue);

    if(peephole) 
        zot += (*ct) * (*Wc)({{2*numUnits, 3*numUnits}});            // add peephole connections to output gate zot + ct*Wc

    // current cell output = ot*activation(ct)   
    auto htNoPeepHole = sigmoid(zot) * activation(*ct);      // = [bS x numUnits]

    // apply projection
    if(projection) {
        ht->assign( mmul(htNoPeepHole, *Wp) );                           // [bS x numUnits] * [ numUnits x numProj] = [bS x numProj]
        // if clipping projection is provided then projected cell output state is clipped by this value 
        if(clippingProjValue != 0.)
            clipping(ht, clippingProjValue);
    }
    else
        ht->assign(&htNoPeepHole);     
}

//////////////////////////////////////////////////////////////////////////

void lstmBlockCell(const NDArray* xt, const NDArray* cLast, const NDArray* yLast,
                   const NDArray* W, const NDArray* Wci, const NDArray* Wcf, const NDArray* Wco, const NDArray* b,
                   const NDArray* z, const NDArray* i, const NDArray* f, const NDArray* o, const NDArray* h, NDArray* c, NDArray* y, const std::vector<double>& params) {

    /* Input arrays:
    *    0: input [bS, inSize] at time t
    *    1: previous cell state  [bS, numUnits], time t-1
    *    2: previous output [bS, numUnits], time t-1
    *    3: Weights - concatenated (input-to-hidden, hidden-to-hidden weights)  weights, [(inSize+numUnits), 4*numUnits]
    *    4: weights - cell peephole (t-1) connections to input modulation gate, [numUnits]
    *    5: weights - cell peephole (t-1) connections to forget gate, [numUnits]
    *    6: weights - cell peephole (t) connections to output gate, [numUnits]
    *    7: biases, [4*numUnits]
    *
    *  Input integer arguments:
    *    0: if not zero, provide peephole connections
    *
    *  Input float arguments:
    *    0: the bias added to forget gates in order to reduce the scale of forgetting in the beginning of the training
    *    1: clipping value for cell state, if it is not equal to zero, then cell state is clipped
    *
    * Output arrays:
    *    0: Output - input gate activations [bs, numUnits]
    *    1: Output - input modulation gate activations [bS, numUnits]
    *    2: Output - forget gate activations [bs, numUnits]
    *    3: Output - output gate activations [bs, numUnits]
    *    4: Activations, pre input gate [bs, numUnits]
    *    5: Activations, cell state [bs, numUnits]
    *    6: Current cell output [bS, numUnits], time t
    */
    nd4j_printf("Start of lstmBlockCell call\n","");
    const bool peephole   = (bool)params[0];        // if true, provide peephole connections
    const double forgetBias    = params[1];
    const double clippingCellValue   = params[2];              // clipping value for ct, if it is not equal to zero, then cell state is clipped


    const int bS   = xt->sizeAt(0);
    const int inSize      = xt->sizeAt(1);
    const int numUnits    = cLast->sizeAt(1);

    //Concat inputs: [xt, yt-1]: concat([bs,nIn],[bs,nOut]) -> [bs, (nIn+nOut)]
    auto concat = new nd4j::ops::concat();
    std::vector<NDArray*> inputs;
    std::vector<double> targs;
    std::vector<Nd4jLong> iargs({1});   //Axis = 1
    std::vector<bool> bargs;
    inputs.emplace_back(const_cast<NDArray*>(xt));
    inputs.emplace_back(const_cast<NDArray*>(yLast));

    nd4j_printf("Before concat call\n","");
    auto result = concat->execute(inputs, targs, iargs, bargs);
    auto concatOut = result->at(0);

    for( int i=0; i<4; i++ ){
        nd4j_printf("First 4 concat elements: <%f>\n", concatOut->e<float>(i));
    }

    nd4j_printf("Before mmul call\n","");
    auto m = mmul(*concatOut, *W);    //mmul: [bs, (nIn+numUnits)]* [(inSize+numUnits), 4*numUnits] = [bs, 4*numUnits]
    nd4j_printf("Before bias add\n","");
    m += (*b);

    for( int i=0; i<4; i++ ){
        nd4j_printf("First 4 mmul elements: <%f>\n", m.e<float>(i));
    }

    auto zz = m({0,0, 0,            numUnits});      	// z for input gate, [bS, numUnits]
    auto zf = m({0,0, numUnits,   2*numUnits});      	// z for forget gate, [bS, numUnits]
    auto zi = m({0,0, 2*numUnits, 3*numUnits});      	// z for input modulation gate, [bS, numUnits]
    auto zo = m({0,0, 3*numUnits, 4*numUnits});      	// z for output gate, [bS, numUnits]

    const_cast<NDArray*>(z)->assign(&zz);
    const_cast<NDArray*>(i)->assign(&zi);
    const_cast<NDArray*>(f)->assign(&zf);
    const_cast<NDArray*>(o)->assign(&zo);


    if(peephole) {                                              // add peephole connections: z  +  ct_1*Wc
        nd4j_printf("Before peepholes\n","");
        zi += (*cLast) * (*Wci);       // add peephole connections to input gate
        zf += (*cLast) * (*Wcf);       // add peephole connections to forget gate
    }

    // current sell state = ft*cLast + it*activation(mmul(Wxc,xt) + mmul(Whc,ht_1) + bc
    if(forgetBias != 0.0){
        nd4j_printf("Before forget bias\n","");
        zf += forgetBias;
    }
    nd4j_printf("Before c assign call\n","");
    c->assign( sigmoid(zf) * (*cLast) + sigmoid(zi) * activation(zz) );

    for( int i=0; i<4; i++ ){
        nd4j_printf("First 4 C elements: <%f>\n", c->e<float>(i));
    }

    // if clipping value is provided then cell state is clipped by this value prior to the cell output activation
    if(clippingCellValue != 0.0) {
        nd4j_printf("Before cell clipping\n","");
        clipping(c, clippingCellValue);
    }

    if(peephole) {
        nd4j_printf("Before peephole 2\n","");
        zo += (*c) * (*Wcf);            // add peephole connections to output gate zot + ct*Wc
    }

    // current cell output = ot*activation(ct)
    nd4j_printf("Before final assign\n","");
    y->assign(sigmoid(zo) * activation(*c));

    for( int i=0; i<4; i++ ){
        nd4j_printf("First 4 y elements: <%f>\n", y->e<float>(i));
    }

    //TODO do I need to delete vairable space and concat op??
}



//////////////////////////////////////////////////////////////////////////
void lstmTimeLoop(const NDArray* x, const NDArray* h0, const NDArray* c0, const NDArray* Wx, const NDArray* Wh, const NDArray* Wc, const NDArray* Wp, const NDArray* b,
                  NDArray* h, NDArray* c, const std::vector<double>& params) {
    
    // x  input [time x bS x inSize]
    // h0 initial cell output (at time step = 0) [bS x numProj], in case of projection=false -> numProj == numUnits !!!
    // c0 initial cell state  (at time step = 0) [bS x numUnits],

    // Wx input-to-hidden  weights, [inSize  x 4*numUnits]
    // Wh hidden-to-hidden weights, [numProj x 4*numUnits]
    // Wc diagonal weights for peephole connections [3*numUnits]
    // Wp projection weights [numUnits x numProj]
    // b  biases, [4*numUnits]
    
    // h cell outputs [time x bS x numProj], that is per each time step
    // c cell states  [time x bS x numUnits] that is per each time step

    const int time  = x->sizeAt(0);

    NDArray currentH(*h0);
    NDArray currentC(*c0);

    // loop through time steps
    for (int t = 0; t < time; ++t) {
        auto xt = (*x)({t,t+1, 0,0, 0,0});
        auto ht = (*h)({t,t+1, 0,0, 0,0});
        auto ct = (*c)({t,t+1, 0,0, 0,0});

        helpers::lstmCell(&xt,&currentH,&currentC, Wx,Wh,Wc,Wp, b,   &ht, &ct,   params);
        currentH.assign(ht);
        currentC.assign(ct);
    }    
}


}
}
}

