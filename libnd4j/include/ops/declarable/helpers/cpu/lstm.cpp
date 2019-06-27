/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
#include <MmulHelper.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
void lstmCell(nd4j::LaunchContext * context, const NDArray* xt, const NDArray* ht_1, const NDArray* ct_1, const NDArray* Wx, const NDArray* Wh, const NDArray* Wc, const NDArray* Wp, const NDArray* b,
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

    // current sell state = ft*ct_1 + it*tanh(mmul(Wxc,xt) + mmul(Whc,ht_1) + bc
    ct->assign( sigmoid(zft + forgetBias) * (*ct_1) + sigmoid(zit) * tanh(zct) );

    // if clipping value is provided then cell state is clipped by this value prior to the cell output activation
    if(clippingCellValue > 0.0)
        clipping(ct, clippingCellValue);

    if(peephole)
        zot += (*ct) * (*Wc)({{2*numUnits, 3*numUnits}});            // add peephole connections to output gate zot + ct*Wc

    // current cell output = ot*tanh(ct)
    auto htNoPeepHole = sigmoid(zot) * tanh(*ct);      // = [bS x numUnits]

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

template <typename T>
static void fusedTanh(NDArray *z, NDArray *i, NDArray *c, const NDArray *cLast, NDArray *f, NDArray *h) {
    //cell state = blockInput .* inputGate + prevCellState .* forgetGate
    /*
    z->applyPairwiseTransform(pairwise::Multiply, i, c, nullptr);       //c = z * i
    auto temp = (*f) * (*cLast);
    *c += temp;                              //c = (i * z) + (zf * (*cLast))
    c->applyTransform(transform::Tanh, h);  //h = tanh(c)
     */

    auto uLen = static_cast<uint>(z->lengthOf());
    auto c_ = c->bufferAsT<T>();
    auto z_ = z->bufferAsT<T>();
    auto i_ = i->bufferAsT<T>();
    auto f_ = f->bufferAsT<T>();
    auto cLast_ = cLast->bufferAsT<T>();
    auto h_ = h->bufferAsT<T>();

    PRAGMA_OMP_PARALLEL_FOR_SIMD
    for (uint e = 0; e < uLen; e++) {
        c_[e] = z_[e] * i_[e] + (f_[e] * cLast_[e]);
        h_[e] = nd4j::math::nd4j_tanh<T,T>(c_[e]);
    }
}

//////////////////////////////////////////////////////////////////////////

void lstmBlockCell(const NDArray* xt, const NDArray* cLast, const NDArray* yLast,
                   const NDArray* W, const NDArray* Wci, const NDArray* Wcf, const NDArray* Wco, const NDArray* b,
                   NDArray* i, NDArray* c, NDArray* f, NDArray* o, NDArray* z, NDArray* h, NDArray* y, const std::vector<double>& params) {

    /* Input arrays:
    *    0: xt              - input [bS, inSize] at time t
    *    1: cLast (cs_prev) - previous cell state  [bS, numUnits], time t-1
    *    2: yLast (h_prev)  - previous output [bS, numUnits], time t-1
    *    3: W               - Weights - concatenated (input-to-hidden, hidden-to-hidden weights)  weights, [(inSize+numUnits), 4*numUnits]
    *    4: Wci             - weights - cell peephole (t-1) connections to input modulation gate, [numUnits]
    *    5: Wcf             - weights - cell peephole (t-1) connections to forget gate, [numUnits]
    *    6: Wco             - weights - cell peephole (t) connections to output gate, [numUnits]
    *    7: b               - biases, [4*numUnits]
    *
    *  Input integer arguments:
    *    0: if not zero, provide peephole connections
    *
    *  Input float arguments:
    *    0: the bias added to forget gates in order to reduce the scale of forgetting in the beginning of the training
    *    1: clipping value for cell state, if it is not equal to zero, then cell state is clipped
    *
    * Output arrays:
    *    0: i      - Input modulation gate activations [bS, numUnits]
    *    1: c (cs) - Cell state (pre tanh) [bs, numUnits] (cs)
    *    2: f      - Output - forget gate activations [bs, numUnits]
    *    3: o      - Output - output gate activations [bs, numUnits]
    *    4: z (ci) - Output - block input [bs, numUnits]
    *    5: h (co) - Cell state, post tanh [bs, numUnits]
    *    6: y (h)  - Current cell output [bS, numUnits], time t
    */
    const bool peephole   = (bool)params[0];        // if true, provide peephole connections
    const double forgetBias    = params[1];
    const double clippingCellValue   = params[2];              // clipping value for ct, if it is not equal to zero, then cell state is clipped


    const int bS   = xt->sizeAt(0);
    const int inSize      = xt->sizeAt(1);
    const int numUnits    = cLast->sizeAt(1);

    //Concat inputs: [xt, yt-1]: concat([bs,nIn],[bs,nOut]) -> [bs, (nIn+nOut)]
    nd4j::ops::concat concat;
    Context cContext(119);
    auto concatOut = NDArrayFactory::create(xt->ordering(), {xt->sizeAt(0), xt->sizeAt(1) + yLast->sizeAt(1)}, xt->dataType(), xt->getContext());
    cContext.setInputArray(0, const_cast<NDArray*>(xt), false);
    cContext.setInputArray(1, const_cast<NDArray*>(yLast), false);
    cContext.setOutputArray(0, &concatOut, false);
    cContext.getIArguments()->emplace_back(1);

    concat.execute(&cContext);

    //NDArray* NDArrayFactory::create_( const char order, const std::vector<Nd4jLong> &shape, nd4j::DataType dataType, nd4j::memory::Workspace* workspace) {
    std::vector<Nd4jLong> shape = {bS, 4*numUnits};
    auto m = NDArrayFactory::create('c', shape, xt->dataType());
    MmulHelper::mmul(&concatOut, W, &m, 1.0f, 0.0f, 'c'); //mmul: [bs, (nIn+numUnits)]* [(inSize+numUnits), 4*numUnits] = [bs, 4*numUnits] - C result array
    m += (*b);  //addiRowVector

    //Note: weights are ordered [inputGate, blockInput, forgetGate, outputGate] to match TF (TF code comments state [i,f,z/ci,o] but behaviour is [i,z,f,o])
    auto zi = (m)({0,0, 0,            numUnits});      	// z for input modulation gate, [bS, numUnits]
    auto zz = (m)({0,0, numUnits, 2*numUnits});      	    // z for block input, [bS, numUnits]
    auto zf = (m)({0,0, 2*numUnits, 3*numUnits});      	// z for forget gate, [bS, numUnits]
    auto zo = (m)({0,0, 3*numUnits, 4*numUnits});      	// z for output gate, [bS, numUnits]

    if(peephole) {                                              // add peephole connections: z  +  ct_1*Wc
        zi += (*cLast) * (*Wci);       // add peephole connections to input gate
        zf += (*cLast) * (*Wcf);       // add peephole connections to forget gate
    }

    // current sell state = ft*cLast + it*tanh(mmul(Wxc,xt) + mmul(Whc,ht_1) + bc
    if(forgetBias != 0.0){
        zf += forgetBias;
    }

    PRAGMA_OMP_PARALLEL
    #pragma omp single
    {
        #pragma omp task
        zz.applyTransform(transform::Tanh, z);      //z = tanh(zz)

        #pragma omp task
        zi.applyTransform(transform::Sigmoid, i);   //i = sigmoid(zi)

        #pragma omp task
        zf.applyTransform(transform::Sigmoid, f);   //f = sigmoid(zf);
    }


    if (z->ews() == 1 && i->ews() == 1 && c->ews() == 1 && cLast->ews() == 1 && f->ews() == 1 && h->ews() == 1 &&
        z->ordering() == i->ordering() && z->ordering() == c->ordering() && z->ordering() == cLast->ordering() && z->ordering() == f->ordering() && z->ordering() == h->ordering()) {
        //cell state = blockInput .* inputGate + prevCellState .* forgetGate
        BUILD_SINGLE_SELECTOR(z->dataType(), fusedTanh, (z, i, c, cLast, f, h), FLOAT_TYPES);
    } else {
        //cell state = blockInput .* inputGate + prevCellState .* forgetGate
        z->applyPairwiseTransform(pairwise::Multiply, i, c, nullptr);       //c = z * i
        auto temp = (*f) * (*cLast);
        *c += temp;                              //c = (i * z) + (zf * (*cLast))
        c->applyTransform(transform::Tanh, h);  //h = tanh(c)
    }

    // if clipping value is provided then cell state is clipped by this value prior to the cell output activation
    if(clippingCellValue > 0.0) {
        clipping(c, clippingCellValue);
    }

    if(peephole) {
        // add peephole connections to output gate zot + ct*Wc
        auto prod = *c * (*Wco);
        zo += prod;
    }
    zo.applyTransform(transform::Sigmoid, o);   // o = sigmoid(zo)

    // current cell output = ot*tanh(ct)
    c->applyTransform(transform::Tanh, h);  //h = tanh(c)
    o->applyPairwiseTransform(pairwise::Multiply, h, y, nullptr);   //y = o * h
}




}
}
}

