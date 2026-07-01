/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.nd4j.linalg.factory.ops;

import static org.nd4j.linalg.factory.NDValidation.isSameType;

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDValidation;

public class NDGNN {
  public NDGNN() {
  }

  /**
   * Approximate Personalized Propagation of Neural Predictions (Klicpera et al. 2019).<br>
   * Decouples prediction from propagation: k steps of personalized PageRank with teleport alpha.<br>
   *
   * @param X Node features [rows, F] (FLOATING_POINT type)
   * @param W Prediction weight [F, H] (FLOATING_POINT type)
   * @param bias Optional bias [H]; pass null to omit (FLOATING_POINT type)
   * @param aNormVals CSR values of the normalised adjacency [nnz] (FLOATING_POINT type)
   * @param aNormColIdx CSR column indices [nnz] INT32 (INT type)
   * @param aNormRowPtr CSR row pointers [rows+1] INT32 (INT type)
   * @param rows Number of nodes
   * @param cols Number of columns of A_norm
   * @param k Number of propagation steps
   * @param alpha Teleport (restart) probability in [0,1)
   * @return out Propagated node predictions [rows, H] (FLOATING_POINT type)
   */
  public INDArray appnp(INDArray X, INDArray W, INDArray bias, INDArray aNormVals,
      INDArray aNormColIdx, INDArray aNormRowPtr, long rows, long cols, int k, double alpha) {
    NDValidation.validateFloatingPoint("appnp", "X", X);
    NDValidation.validateFloatingPoint("appnp", "W", W);
    NDValidation.validateFloatingPoint("appnp", "bias", bias);
    NDValidation.validateFloatingPoint("appnp", "aNormVals", aNormVals);
    NDValidation.validateInteger("appnp", "aNormColIdx", aNormColIdx);
    NDValidation.validateInteger("appnp", "aNormRowPtr", aNormRowPtr);
    INDArray H0 = org.nd4j.linalg.factory.Nd4j.base().mmul(X, W);
    if (bias != null) {
        H0 = H0.add(bias);
    }
    INDArray H = H0;
    for (int i = 0; i < k; i++) {
        INDArray AH = org.nd4j.linalg.factory.Nd4j.sparse().csrSpmm(aNormVals, aNormColIdx, aNormRowPtr, H, (int) rows, (int) cols, false);
        H = AH.mul(1.0 - alpha).add(H0.mul(alpha));
    }
    INDArray out = H;
    return out;
  }

  /**
   * Chebyshev spectral graph convolution (Defferrard et al. 2016).<br>
   * T_0=X, T_1=L_hat*X, T_k=2*L_hat*T_{k-1}-T_{k-2}; out=sum_k T_k*W_k<br>
   *
   * @param X Node features [rows, F] (FLOATING_POINT type)
   * @param weights K Chebyshev-coefficient matrices, each [F, H] (FLOATING_POINT type)
   * @param lapVals CSR values of the scaled Laplacian L_hat [nnz] (FLOATING_POINT type)
   * @param colIdx CSR column indices [nnz] INT32 (INT type)
   * @param rowPtr CSR row pointers [rows+1] INT32 (INT type)
   * @param rows Number of nodes
   * @param cols Number of columns of L_hat
   * @return out Filtered node embeddings [rows, H] (FLOATING_POINT type)
   */
  public INDArray chebConv(INDArray X, INDArray[] weights, INDArray lapVals, INDArray colIdx,
      INDArray rowPtr, long rows, long cols) {
    NDValidation.validateFloatingPoint("chebConv", "X", X);
    NDValidation.validateFloatingPoint("chebConv", "weights", weights);
    Preconditions.checkArgument(weights.length >= 1, "weights has incorrect size/length. Expected: weights.length >= 1, got %s", weights.length);
    NDValidation.validateFloatingPoint("chebConv", "lapVals", lapVals);
    NDValidation.validateInteger("chebConv", "colIdx", colIdx);
    NDValidation.validateInteger("chebConv", "rowPtr", rowPtr);
    INDArray tPrev2 = X;
    INDArray out = org.nd4j.linalg.factory.Nd4j.base().mmul(X, weights[0]);
    if (weights.length >= 2) {
        INDArray tPrev1 = org.nd4j.linalg.factory.Nd4j.sparse().csrSpmm(lapVals, colIdx, rowPtr, X, (int) rows, (int) cols, false);
        out = out.add(org.nd4j.linalg.factory.Nd4j.base().mmul(tPrev1, weights[1]));
        for (int k = 2; k < weights.length; k++) {
            INDArray lTk = org.nd4j.linalg.factory.Nd4j.sparse().csrSpmm(lapVals, colIdx, rowPtr, tPrev1, (int) rows, (int) cols, false);
            INDArray tk = lTk.mul(2.0).sub(tPrev2);
            out = out.add(org.nd4j.linalg.factory.Nd4j.base().mmul(tk, weights[k]));
            tPrev2 = tPrev1;
            tPrev1 = tk;
        }
    }
    return out;
  }

  /**
   * CompGCN convolution (Vashishth et al. 2020): multi-relational GNN composing entity+relation embeddings.<br>
   * compOp=0: sub (TransE-style); else: elementwise mult (DistMult-style).<br>
   *
   * @param X Entity embeddings [n, dim] (FLOATING_POINT type)
   * @param relEmb Relation embeddings [numRelations, dim] (FLOATING_POINT type)
   * @param edgeRelIdx Per-edge relation index [nnz] INT32 (INT type)
   * @param W Output weight [dim, dim_out] (FLOATING_POINT type)
   * @param colIdx CSR column indices [nnz] INT32 (INT type)
   * @param rowPtr CSR row pointers [rows+1] INT32 (INT type)
   * @param rows Number of entities (nodes)
   * @param n Number of entities (for CsrEdgeGather doDiff)
   * @param compOp Composition: 0=subtraction (TransE-style), else element-wise mult (DistMult-style)
   * @param applyRelu Whether to apply ReLU to the output
   * @return out Updated entity embeddings [rows, dim_out] (FLOATING_POINT type)
   */
  public INDArray compGcnConv(INDArray X, INDArray relEmb, INDArray edgeRelIdx, INDArray W,
      INDArray colIdx, INDArray rowPtr, long rows, long n, int compOp, boolean applyRelu) {
    NDValidation.validateFloatingPoint("compGcnConv", "X", X);
    NDValidation.validateFloatingPoint("compGcnConv", "relEmb", relEmb);
    NDValidation.validateInteger("compGcnConv", "edgeRelIdx", edgeRelIdx);
    NDValidation.validateFloatingPoint("compGcnConv", "W", W);
    NDValidation.validateInteger("compGcnConv", "colIdx", colIdx);
    NDValidation.validateInteger("compGcnConv", "rowPtr", rowPtr);
    INDArray Xj = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeGather(colIdx, X, (int) n);
    INDArray Rj = org.nd4j.linalg.factory.Nd4j.base().gather(relEmb, edgeRelIdx, 0);
    INDArray phi = (compOp == 0) ? Xj.sub(Rj) : Xj.mul(Rj);
    INDArray agg = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeAggregate(rowPtr, phi, (int) rows, 1);
    INDArray out = org.nd4j.linalg.factory.Nd4j.base().mmul(agg, W);
    if (applyRelu) {
        out = org.nd4j.linalg.factory.Nd4j.nn().relu(out, 0.0);
    }
    return out;
  }

  /**
   * Single-head Graph Attention Network convolution (Veličković et al. 2018).<br>
   *
   * @param X Node features [N, F] (FLOATING_POINT type)
   * @param W Linear weight [F, H] (FLOATING_POINT type)
   * @param attSrc Attention source vector [H, 1] (FLOATING_POINT type)
   * @param attDst Attention destination vector [H, 1] (FLOATING_POINT type)
   * @param colIdx CSR column indices (source nodes per edge) [nnz] INT32 (INT type)
   * @param rowPtr CSR row pointers [N+1] INT32 (INT type)
   * @param rowIdx Per-edge destination-node index [nnz] INT32 (INT type)
   * @param nnz Number of edges
   * @param N Number of nodes
   * @param leakyAlpha Negative slope for LeakyReLU
   * @return out Aggregated node embeddings [N, H] (FLOATING_POINT type)
   */
  public INDArray gatConvHead(INDArray X, INDArray W, INDArray attSrc, INDArray attDst,
      INDArray colIdx, INDArray rowPtr, INDArray rowIdx, long nnz, long N, double leakyAlpha) {
    NDValidation.validateFloatingPoint("gatConvHead", "X", X);
    NDValidation.validateFloatingPoint("gatConvHead", "W", W);
    NDValidation.validateFloatingPoint("gatConvHead", "attSrc", attSrc);
    NDValidation.validateFloatingPoint("gatConvHead", "attDst", attDst);
    NDValidation.validateInteger("gatConvHead", "colIdx", colIdx);
    NDValidation.validateInteger("gatConvHead", "rowPtr", rowPtr);
    NDValidation.validateInteger("gatConvHead", "rowIdx", rowIdx);
    INDArray Wh = org.nd4j.linalg.factory.Nd4j.base().mmul(X, W);
    INDArray srcFeat = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeGather(colIdx, Wh, (int) N);
    INDArray dstFeat = org.nd4j.linalg.factory.Nd4j.base().gather(Wh, rowIdx, 0);
    INDArray eSrc = org.nd4j.linalg.factory.Nd4j.base().reshape(org.nd4j.linalg.factory.Nd4j.base().mmul(srcFeat, attSrc), nnz);
    INDArray eDst = org.nd4j.linalg.factory.Nd4j.base().reshape(org.nd4j.linalg.factory.Nd4j.base().mmul(dstFeat, attDst), nnz);
    INDArray eLogit = org.nd4j.linalg.factory.Nd4j.nn().leakyRelu(eSrc.add(eDst), leakyAlpha);
    INDArray alpha = org.nd4j.linalg.factory.Nd4j.sparse().csrRowSoftmax(eLogit, rowPtr, (int) N);
    INDArray weighted = srcFeat.mul(org.nd4j.linalg.factory.Nd4j.base().reshape(alpha, nnz, 1));
    INDArray out = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeAggregate(rowPtr, weighted, (int) N, 0);
    return out;
  }

  /**
   * Single-head GATv2 convolution (Brody et al. 2021). GATv2 applies the nonlinearity before<br>
   * the attention projection (dynamic attention), fixing the static attention limitation of GAT v1.<br>
   *
   * @param X Node features [N, F] (FLOATING_POINT type)
   * @param W Linear weight [F, H] (FLOATING_POINT type)
   * @param att Shared attention vector [H, 1] (FLOATING_POINT type)
   * @param colIdx CSR column indices (source nodes per edge) [nnz] INT32 (INT type)
   * @param rowPtr CSR row pointers [N+1] INT32 (INT type)
   * @param rowIdx Per-edge destination-node index [nnz] INT32 (INT type)
   * @param nnz Number of edges
   * @param N Number of nodes
   * @param leakyAlpha Negative slope for LeakyReLU
   * @return out Aggregated node embeddings [N, H] (FLOATING_POINT type)
   */
  public INDArray gatV2ConvHead(INDArray X, INDArray W, INDArray att, INDArray colIdx,
      INDArray rowPtr, INDArray rowIdx, long nnz, long N, double leakyAlpha) {
    NDValidation.validateFloatingPoint("gatV2ConvHead", "X", X);
    NDValidation.validateFloatingPoint("gatV2ConvHead", "W", W);
    NDValidation.validateFloatingPoint("gatV2ConvHead", "att", att);
    NDValidation.validateInteger("gatV2ConvHead", "colIdx", colIdx);
    NDValidation.validateInteger("gatV2ConvHead", "rowPtr", rowPtr);
    NDValidation.validateInteger("gatV2ConvHead", "rowIdx", rowIdx);
    INDArray Wh = org.nd4j.linalg.factory.Nd4j.base().mmul(X, W);
    INDArray srcFeat = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeGather(colIdx, Wh, (int) N);
    INDArray dstFeat = org.nd4j.linalg.factory.Nd4j.base().gather(Wh, rowIdx, 0);
    INDArray g = org.nd4j.linalg.factory.Nd4j.nn().leakyRelu(srcFeat.add(dstFeat), leakyAlpha);
    INDArray eLogit = org.nd4j.linalg.factory.Nd4j.base().reshape(org.nd4j.linalg.factory.Nd4j.base().mmul(g, att), nnz);
    INDArray alpha = org.nd4j.linalg.factory.Nd4j.sparse().csrRowSoftmax(eLogit, rowPtr, (int) N);
    INDArray weighted = srcFeat.mul(org.nd4j.linalg.factory.Nd4j.base().reshape(alpha, nnz, 1));
    INDArray out = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeAggregate(rowPtr, weighted, (int) N, 0);
    return out;
  }

  /**
   * Graph Convolutional Network layer (Kipf & Welling 2017).<br>
   * out = relu?( A_norm · X · W + bias )<br>
   *
   * @param X Node features [rows, F] (FLOATING_POINT type)
   * @param W Weight matrix [F, H] (FLOATING_POINT type)
   * @param bias Optional bias [H]; pass null to omit (FLOATING_POINT type)
   * @param aNormVals CSR non-zero values of the normalised adjacency [nnz] (FLOATING_POINT type)
   * @param aNormColIdx CSR column indices [nnz] (INT type)
   * @param aNormRowPtr CSR row pointers [rows+1] (INT type)
   * @param rows Number of nodes / rows in A_norm
   * @param cols Number of columns in A_norm
   * @param applyRelu Whether to apply ReLU to the output
   * @return out Output node embeddings [rows, H] (FLOATING_POINT type)
   */
  public INDArray gcnConv(INDArray X, INDArray W, INDArray bias, INDArray aNormVals,
      INDArray aNormColIdx, INDArray aNormRowPtr, long rows, long cols, boolean applyRelu) {
    NDValidation.validateFloatingPoint("gcnConv", "X", X);
    NDValidation.validateFloatingPoint("gcnConv", "W", W);
    NDValidation.validateFloatingPoint("gcnConv", "bias", bias);
    NDValidation.validateFloatingPoint("gcnConv", "aNormVals", aNormVals);
    NDValidation.validateInteger("gcnConv", "aNormColIdx", aNormColIdx);
    NDValidation.validateInteger("gcnConv", "aNormRowPtr", aNormRowPtr);
    INDArray AX = org.nd4j.linalg.factory.Nd4j.sparse().csrSpmm(aNormVals, aNormColIdx, aNormRowPtr, X, (int) rows, (int) cols, false);
    INDArray AXW = org.nd4j.linalg.factory.Nd4j.base().mmul(AX, W);
    if (bias != null) {
        AXW = AXW.add(bias);
    }
    INDArray out = applyRelu ? org.nd4j.linalg.factory.Nd4j.nn().relu(AXW, 0.0) : AXW;
    return out;
  }

  /**
   * GCNII convolution (Chen et al. 2020): deep GCN layer combining initial residual connection<br>
   * with identity mapping. M = (1-alpha)*(A_norm*H) + alpha*H0; out = sigma((1-beta)*M + beta*(M*W))<br>
   *
   * @param H Current layer representation [rows, F] (FLOATING_POINT type)
   * @param H0 Initial (input-projected) representation [rows, F] (FLOATING_POINT type)
   * @param W Weight matrix [F, F] (FLOATING_POINT type)
   * @param aNormVals CSR values of the normalised adjacency [nnz] (FLOATING_POINT type)
   * @param aNormColIdx CSR column indices [nnz] INT32 (INT type)
   * @param aNormRowPtr CSR row pointers [rows+1] INT32 (INT type)
   * @param rows Number of nodes
   * @param cols Number of columns of A_norm
   * @param alpha Initial-residual strength in [0,1]
   * @param beta Identity-mapping strength in [0,1]
   * @param applyRelu Whether to apply ReLU to the output
   * @return out Updated node representation [rows, F] (FLOATING_POINT type)
   */
  public INDArray gcniiConv(INDArray H, INDArray H0, INDArray W, INDArray aNormVals,
      INDArray aNormColIdx, INDArray aNormRowPtr, long rows, long cols, double alpha, double beta,
      boolean applyRelu) {
    NDValidation.validateFloatingPoint("gcniiConv", "H", H);
    NDValidation.validateFloatingPoint("gcniiConv", "H0", H0);
    NDValidation.validateFloatingPoint("gcniiConv", "W", W);
    NDValidation.validateFloatingPoint("gcniiConv", "aNormVals", aNormVals);
    NDValidation.validateInteger("gcniiConv", "aNormColIdx", aNormColIdx);
    NDValidation.validateInteger("gcniiConv", "aNormRowPtr", aNormRowPtr);
    INDArray AH = org.nd4j.linalg.factory.Nd4j.sparse().csrSpmm(aNormVals, aNormColIdx, aNormRowPtr, H, (int) rows, (int) cols, false);
    INDArray M = AH.mul(1.0 - alpha).add(H0.mul(alpha));
    INDArray MW = org.nd4j.linalg.factory.Nd4j.base().mmul(M, W);
    INDArray out = M.mul(1.0 - beta).add(MW.mul(beta));
    if (applyRelu) {
        out = org.nd4j.linalg.factory.Nd4j.nn().relu(out, 0.0);
    }
    return out;
  }

  /**
   * Gated Graph Neural Network (Li et al. 2016): steps rounds of neighbour aggregation + GRU update.<br>
   * Uses concat([in1,in2])*concat([W1,W2]) gate form for correct CUDA backward.<br>
   *
   * @param X Initial node states [rows, H] (FLOATING_POINT type)
   * @param aggW Message transform [H, H] (FLOATING_POINT type)
   * @param wz Update-gate weight [H, H] (FLOATING_POINT type)
   * @param uz Update-gate recurrent weight [H, H] (FLOATING_POINT type)
   * @param wr Reset-gate weight [H, H] (FLOATING_POINT type)
   * @param ur Reset-gate recurrent weight [H, H] (FLOATING_POINT type)
   * @param wh Candidate weight [H, H] (FLOATING_POINT type)
   * @param uh Candidate recurrent weight [H, H] (FLOATING_POINT type)
   * @param aNormVals CSR values of the normalised adjacency [nnz] (FLOATING_POINT type)
   * @param aNormColIdx CSR column indices [nnz] INT32 (INT type)
   * @param aNormRowPtr CSR row pointers [rows+1] INT32 (INT type)
   * @param rows Number of nodes
   * @param cols Columns of A_norm
   * @param steps Number of propagation / GRU steps
   * @return out Final node states [rows, H] (FLOATING_POINT type)
   */
  public INDArray ggnn(INDArray X, INDArray aggW, INDArray wz, INDArray uz, INDArray wr,
      INDArray ur, INDArray wh, INDArray uh, INDArray aNormVals, INDArray aNormColIdx,
      INDArray aNormRowPtr, long rows, long cols, int steps) {
    NDValidation.validateFloatingPoint("ggnn", "X", X);
    NDValidation.validateFloatingPoint("ggnn", "aggW", aggW);
    NDValidation.validateFloatingPoint("ggnn", "wz", wz);
    NDValidation.validateFloatingPoint("ggnn", "uz", uz);
    NDValidation.validateFloatingPoint("ggnn", "wr", wr);
    NDValidation.validateFloatingPoint("ggnn", "ur", ur);
    NDValidation.validateFloatingPoint("ggnn", "wh", wh);
    NDValidation.validateFloatingPoint("ggnn", "uh", uh);
    NDValidation.validateFloatingPoint("ggnn", "aNormVals", aNormVals);
    NDValidation.validateInteger("ggnn", "aNormColIdx", aNormColIdx);
    NDValidation.validateInteger("ggnn", "aNormRowPtr", aNormRowPtr);
    INDArray h = X;
    for (int t = 0; t < steps; t++) {
        INDArray agg = org.nd4j.linalg.factory.Nd4j.sparse().csrSpmm(aNormVals, aNormColIdx, aNormRowPtr, h, (int) rows, (int) cols, false);
        INDArray a = org.nd4j.linalg.factory.Nd4j.base().mmul(agg, aggW);
        INDArray ah = org.nd4j.linalg.factory.Nd4j.base().concat(1, a, h);
        INDArray z = org.nd4j.linalg.factory.Nd4j.nn().sigmoid(org.nd4j.linalg.factory.Nd4j.base().mmul(ah, org.nd4j.linalg.factory.Nd4j.base().concat(0, wz, uz)));
        INDArray r = org.nd4j.linalg.factory.Nd4j.nn().sigmoid(org.nd4j.linalg.factory.Nd4j.base().mmul(ah, org.nd4j.linalg.factory.Nd4j.base().concat(0, wr, ur)));
        INDArray arh = org.nd4j.linalg.factory.Nd4j.base().concat(1, a, r.mul(h));
        INDArray hh = org.nd4j.linalg.factory.Nd4j.math().tanh(org.nd4j.linalg.factory.Nd4j.base().mmul(arh, org.nd4j.linalg.factory.Nd4j.base().concat(0, wh, uh)));
        h = h.mul(z.mul(-1.0).add(1.0)).add(z.mul(hh));
    }
    INDArray out = h;
    return out;
  }

  /**
   * Graph Isomorphism Network convolution (Xu et al. 2019) with optional Layer Normalisation.<br>
   *
   * @param X Node features [rows, F] (FLOATING_POINT type)
   * @param w1 First MLP weight [F, H] (FLOATING_POINT type)
   * @param b1 First MLP bias [H] (FLOATING_POINT type)
   * @param w2 Second MLP weight [H, out] (FLOATING_POINT type)
   * @param b2 Second MLP bias [out] (FLOATING_POINT type)
   * @param eps Learnable epsilon scalar SDVariable (shape []) (FLOATING_POINT type)
   * @param colIdx CSR column indices [nnz] INT32 (INT type)
   * @param rowPtr CSR row pointers [rows+1] INT32 (INT type)
   * @param rows Number of nodes
   * @param n Number of nodes (for CsrEdgeGather doDiff)
   * @param layerNorm Whether to apply parameter-free layer normalisation
   * @return out Node embeddings [rows, out], optionally layer-normalised (FLOATING_POINT type)
   */
  public INDArray ginConv(INDArray X, INDArray w1, INDArray b1, INDArray w2, INDArray b2,
      INDArray eps, INDArray colIdx, INDArray rowPtr, long rows, long n, boolean layerNorm) {
    NDValidation.validateFloatingPoint("ginConv", "X", X);
    NDValidation.validateFloatingPoint("ginConv", "w1", w1);
    NDValidation.validateFloatingPoint("ginConv", "b1", b1);
    NDValidation.validateFloatingPoint("ginConv", "w2", w2);
    NDValidation.validateFloatingPoint("ginConv", "b2", b2);
    NDValidation.validateFloatingPoint("ginConv", "eps", eps);
    NDValidation.validateInteger("ginConv", "colIdx", colIdx);
    NDValidation.validateInteger("ginConv", "rowPtr", rowPtr);
    INDArray Xj = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeGather(colIdx, X, (int) n);
    INDArray agg = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeAggregate(rowPtr, Xj, (int) rows, 0);
    INDArray hi = X.mul(eps.add(1.0)).add(agg);
    INDArray h1 = org.nd4j.linalg.factory.Nd4j.nn().relu(org.nd4j.linalg.factory.Nd4j.base().mmul(hi, w1).add(b1), 0.0);
    INDArray out = org.nd4j.linalg.factory.Nd4j.base().mmul(h1, w2).add(b2);
    if (layerNorm) {
        INDArray lnMean = org.nd4j.linalg.factory.Nd4j.base().mean(out, true, 1);
        INDArray d = out.sub(lnMean);
        INDArray variance = org.nd4j.linalg.factory.Nd4j.base().mean(d.mul(d), true, 1);
        out = d.div(org.nd4j.linalg.factory.Nd4j.math().sqrt(variance.add(1e-5)));
    }
    return out;
  }

  /**
   * Graph Isomorphism Network convolution (Xu et al. 2019) with optional Layer Normalisation.<br>
   *
   * @param X Node features [rows, F] (FLOATING_POINT type)
   * @param w1 First MLP weight [F, H] (FLOATING_POINT type)
   * @param b1 First MLP bias [H] (FLOATING_POINT type)
   * @param w2 Second MLP weight [H, out] (FLOATING_POINT type)
   * @param b2 Second MLP bias [out] (FLOATING_POINT type)
   * @param eps Learnable epsilon scalar SDVariable (shape []) (FLOATING_POINT type)
   * @param colIdx CSR column indices [nnz] INT32 (INT type)
   * @param rowPtr CSR row pointers [rows+1] INT32 (INT type)
   * @param rows Number of nodes
   * @param n Number of nodes (for CsrEdgeGather doDiff)
   * @return out Node embeddings [rows, out], optionally layer-normalised (FLOATING_POINT type)
   */
  public INDArray ginConv(INDArray X, INDArray w1, INDArray b1, INDArray w2, INDArray b2,
      INDArray eps, INDArray colIdx, INDArray rowPtr, long rows, long n) {
    NDValidation.validateFloatingPoint("ginConv", "X", X);
    NDValidation.validateFloatingPoint("ginConv", "w1", w1);
    NDValidation.validateFloatingPoint("ginConv", "b1", b1);
    NDValidation.validateFloatingPoint("ginConv", "w2", w2);
    NDValidation.validateFloatingPoint("ginConv", "b2", b2);
    NDValidation.validateFloatingPoint("ginConv", "eps", eps);
    NDValidation.validateInteger("ginConv", "colIdx", colIdx);
    NDValidation.validateInteger("ginConv", "rowPtr", rowPtr);
    boolean layerNorm = false;
    INDArray Xj = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeGather(colIdx, X, (int) n);
    INDArray agg = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeAggregate(rowPtr, Xj, (int) rows, 0);
    INDArray hi = X.mul(eps.add(1.0)).add(agg);
    INDArray h1 = org.nd4j.linalg.factory.Nd4j.nn().relu(org.nd4j.linalg.factory.Nd4j.base().mmul(hi, w1).add(b1), 0.0);
    INDArray out = org.nd4j.linalg.factory.Nd4j.base().mmul(h1, w2).add(b2);
    if (layerNorm) {
        INDArray lnMean = org.nd4j.linalg.factory.Nd4j.base().mean(out, true, 1);
        INDArray d = out.sub(lnMean);
        INDArray variance = org.nd4j.linalg.factory.Nd4j.base().mean(d.mul(d), true, 1);
        out = d.div(org.nd4j.linalg.factory.Nd4j.math().sqrt(variance.add(1e-5)));
    }
    return out;
  }

  /**
   * GraphNorm (Cai et al. 2021): learnable graph-level normalisation. Implemented in transposed<br>
   * [F, rows] layout for correct CUDA gradient flow on per-feature alpha scaling.<br>
   *
   * @param X Node features [rows, F] (FLOATING_POINT type)
   * @param gamma Learnable scale [F] (FLOATING_POINT type)
   * @param beta Learnable shift [F] (FLOATING_POINT type)
   * @param alpha Learnable mean-retention coefficient [F] (FLOATING_POINT type)
   * @return out Normalised node features [rows, F] (FLOATING_POINT type)
   */
  public INDArray graphNorm(INDArray X, INDArray gamma, INDArray beta, INDArray alpha) {
    NDValidation.validateFloatingPoint("graphNorm", "X", X);
    NDValidation.validateFloatingPoint("graphNorm", "gamma", gamma);
    NDValidation.validateFloatingPoint("graphNorm", "beta", beta);
    NDValidation.validateFloatingPoint("graphNorm", "alpha", alpha);
    INDArray Xt = org.nd4j.linalg.factory.Nd4j.base().transpose(X);
    INDArray meanT = org.nd4j.linalg.factory.Nd4j.base().mean(Xt, true, 1);
    INDArray alphaCol = org.nd4j.linalg.factory.Nd4j.base().reshape(alpha, -1, 1);
    INDArray shiftedT = Xt.sub(meanT.mul(alphaCol));
    INDArray varT = org.nd4j.linalg.factory.Nd4j.base().mean(shiftedT.mul(shiftedT), true, 1);
    INDArray normT = shiftedT.div(org.nd4j.linalg.factory.Nd4j.math().sqrt(varT.add(1e-5)));
    INDArray gammaCol = org.nd4j.linalg.factory.Nd4j.base().reshape(gamma, -1, 1);
    INDArray betaCol = org.nd4j.linalg.factory.Nd4j.base().reshape(beta, -1, 1);
    INDArray outT = normT.mul(gammaCol).add(betaCol);
    INDArray out = org.nd4j.linalg.factory.Nd4j.base().transpose(outT);
    return out;
  }

  /**
   * Single-head Graph Transformer layer (Dwivedi & Bresson 2021, simplified):<br>
   * scaled dot-product self-attention, optionally restricted via additive mask.<br>
   *
   * @param X Node features [N, F] (FLOATING_POINT type)
   * @param wq Query weight [F, d] (FLOATING_POINT type)
   * @param wk Key weight [F, d] (FLOATING_POINT type)
   * @param wv Value weight [F, d] (FLOATING_POINT type)
   * @param wo Output weight [d, d_out] (FLOATING_POINT type)
   * @param adjMask Optional additive attention mask [N, N]; pass null for full attention (FLOATING_POINT type)
   * @param scale Score scaling (typically 1/sqrt(d))
   * @return out Node embeddings [N, d_out] (FLOATING_POINT type)
   */
  public INDArray graphTransformer(INDArray X, INDArray wq, INDArray wk, INDArray wv, INDArray wo,
      INDArray adjMask, double scale) {
    NDValidation.validateFloatingPoint("graphTransformer", "X", X);
    NDValidation.validateFloatingPoint("graphTransformer", "wq", wq);
    NDValidation.validateFloatingPoint("graphTransformer", "wk", wk);
    NDValidation.validateFloatingPoint("graphTransformer", "wv", wv);
    NDValidation.validateFloatingPoint("graphTransformer", "wo", wo);
    NDValidation.validateFloatingPoint("graphTransformer", "adjMask", adjMask);
    INDArray q = org.nd4j.linalg.factory.Nd4j.base().mmul(X, wq);
    INDArray k = org.nd4j.linalg.factory.Nd4j.base().mmul(X, wk);
    INDArray v = org.nd4j.linalg.factory.Nd4j.base().mmul(X, wv);
    INDArray scores = org.nd4j.linalg.factory.Nd4j.base().mmul(q, k, false, true, false).mul(scale);
    if (adjMask != null) {
        scores = scores.add(adjMask);
    }
    INDArray attn = org.nd4j.linalg.factory.Nd4j.nn().softmax(scores, 1);
    INDArray ctx = org.nd4j.linalg.factory.Nd4j.base().mmul(attn, v);
    INDArray out = org.nd4j.linalg.factory.Nd4j.base().mmul(ctx, wo);
    return out;
  }

  /**
   * Heterogeneous Attention Network (Wang et al. 2019).<br>
   * For each meta-path: run a single-head node-level GAT (gatConvHead), then compute a semantic<br>
   * attention score via tanh(Z*semW + semB)*semQ. Softmax over meta-path scores gives mixing<br>
   * weights beta_p; final output = sum_p(beta_p * Z_p).<br>
   *
   * @param X Node features [N, F] (FLOATING_POINT type)
   * @param metaW Per-meta-path linear weight [F, H], one per meta-path (FLOATING_POINT type)
   * @param attSrc Per-meta-path attention source vector [H, 1] (FLOATING_POINT type)
   * @param attDst Per-meta-path attention destination vector [H, 1] (FLOATING_POINT type)
   * @param colIdx Per-meta-path CSR column indices [nnz_p] INT32 (INT type)
   * @param rowPtr Per-meta-path CSR row pointers [N+1] INT32 (INT type)
   * @param rowIdx Per-meta-path per-edge destination indices [nnz_p] INT32 (INT type)
   * @param semW Semantic-attention weight [H, A] (FLOATING_POINT type)
   * @param semB Semantic-attention bias [A] (FLOATING_POINT type)
   * @param semQ Semantic-attention query [A, 1] (FLOATING_POINT type)
   * @param nnz Per-meta-path edge counts (Size: AtLeast(min=1))
   * @param N Number of nodes
   * @param leakyAlpha LeakyReLU negative slope for node-level GAT
   * @return out Fused node embeddings [N, H] (FLOATING_POINT type)
   */
  public INDArray han(INDArray X, INDArray[] metaW, INDArray[] attSrc, INDArray[] attDst,
      INDArray[] colIdx, INDArray[] rowPtr, INDArray[] rowIdx, INDArray semW, INDArray semB,
      INDArray semQ, long[] nnz, long N, double leakyAlpha) {
    NDValidation.validateFloatingPoint("han", "X", X);
    NDValidation.validateFloatingPoint("han", "metaW", metaW);
    Preconditions.checkArgument(metaW.length >= 1, "metaW has incorrect size/length. Expected: metaW.length >= 1, got %s", metaW.length);
    NDValidation.validateFloatingPoint("han", "attSrc", attSrc);
    Preconditions.checkArgument(attSrc.length >= 1, "attSrc has incorrect size/length. Expected: attSrc.length >= 1, got %s", attSrc.length);
    NDValidation.validateFloatingPoint("han", "attDst", attDst);
    Preconditions.checkArgument(attDst.length >= 1, "attDst has incorrect size/length. Expected: attDst.length >= 1, got %s", attDst.length);
    NDValidation.validateInteger("han", "colIdx", colIdx);
    Preconditions.checkArgument(colIdx.length >= 1, "colIdx has incorrect size/length. Expected: colIdx.length >= 1, got %s", colIdx.length);
    NDValidation.validateInteger("han", "rowPtr", rowPtr);
    Preconditions.checkArgument(rowPtr.length >= 1, "rowPtr has incorrect size/length. Expected: rowPtr.length >= 1, got %s", rowPtr.length);
    NDValidation.validateInteger("han", "rowIdx", rowIdx);
    Preconditions.checkArgument(rowIdx.length >= 1, "rowIdx has incorrect size/length. Expected: rowIdx.length >= 1, got %s", rowIdx.length);
    NDValidation.validateFloatingPoint("han", "semW", semW);
    NDValidation.validateFloatingPoint("han", "semB", semB);
    NDValidation.validateFloatingPoint("han", "semQ", semQ);
    Preconditions.checkArgument(nnz.length >= 1, "nnz has incorrect size/length. Expected: nnz.length >= 1, got %s", nnz.length);
    final int P = metaW.length;
    INDArray[] z = new INDArray[P];
    INDArray[] expw = new INDArray[P];
    INDArray sumExp = null;
    for (int p = 0; p < P; p++) {
        z[p] = org.nd4j.linalg.factory.Nd4j.gnn().gatConvHead(X, metaW[p], attSrc[p], attDst[p],
                colIdx[p], rowPtr[p], rowIdx[p], nnz[p], N, leakyAlpha);
        INDArray proj  = org.nd4j.linalg.factory.Nd4j.math().tanh(org.nd4j.linalg.factory.Nd4j.base().mmul(z[p], semW).add(semB));
        INDArray score = org.nd4j.linalg.factory.Nd4j.base().mean(org.nd4j.linalg.factory.Nd4j.base().mmul(proj, semQ), false);
        expw[p] = org.nd4j.linalg.factory.Nd4j.math().exp(score);
        sumExp = (p == 0) ? expw[p] : sumExp.add(expw[p]);
    }
    INDArray out = null;
    for (int p = 0; p < P; p++) {
        INDArray betaP = expw[p].div(sumExp);
        INDArray term  = z[p].mul(betaP);
        out = (p == 0) ? term : out.add(term);
    }
    return out;
  }

  /**
   * Simplified single-head Heterogeneous Graph Transformer (Hu et al. 2020):<br>
   * scaled dot-product attention where relation embedding modulates the key.<br>
   *
   * @param X Node features [N, F] (FLOATING_POINT type)
   * @param wq Query weight [F, d] (FLOATING_POINT type)
   * @param wk Key weight [F, d] (FLOATING_POINT type)
   * @param wv Value weight [F, d] (FLOATING_POINT type)
   * @param relEmb Relation embeddings [numRelations, d] (FLOATING_POINT type)
   * @param edgeRelIdx Per-edge relation index [nnz] INT32 (INT type)
   * @param colIdx CSR column indices (source nodes) [nnz] INT32 (INT type)
   * @param rowPtr CSR row pointers [N+1] INT32 (INT type)
   * @param rowIdx Per-edge destination index [nnz] INT32 (INT type)
   * @param nnz Number of edges
   * @param N Number of nodes
   * @param scale Attention scaling (typically 1/sqrt(d))
   * @return out Aggregated node embeddings [N, d] (FLOATING_POINT type)
   */
  public INDArray hgtConvHead(INDArray X, INDArray wq, INDArray wk, INDArray wv, INDArray relEmb,
      INDArray edgeRelIdx, INDArray colIdx, INDArray rowPtr, INDArray rowIdx, long nnz, long N,
      double scale) {
    NDValidation.validateFloatingPoint("hgtConvHead", "X", X);
    NDValidation.validateFloatingPoint("hgtConvHead", "wq", wq);
    NDValidation.validateFloatingPoint("hgtConvHead", "wk", wk);
    NDValidation.validateFloatingPoint("hgtConvHead", "wv", wv);
    NDValidation.validateFloatingPoint("hgtConvHead", "relEmb", relEmb);
    NDValidation.validateInteger("hgtConvHead", "edgeRelIdx", edgeRelIdx);
    NDValidation.validateInteger("hgtConvHead", "colIdx", colIdx);
    NDValidation.validateInteger("hgtConvHead", "rowPtr", rowPtr);
    NDValidation.validateInteger("hgtConvHead", "rowIdx", rowIdx);
    INDArray q = org.nd4j.linalg.factory.Nd4j.base().mmul(X, wq);
    INDArray k = org.nd4j.linalg.factory.Nd4j.base().mmul(X, wk);
    INDArray v = org.nd4j.linalg.factory.Nd4j.base().mmul(X, wv);
    INDArray qDst = org.nd4j.linalg.factory.Nd4j.base().gather(q, rowIdx, 0);
    INDArray kSrc = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeGather(colIdx, k, (int) N);
    INDArray vSrc = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeGather(colIdx, v, (int) N);
    INDArray rel = org.nd4j.linalg.factory.Nd4j.base().gather(relEmb, edgeRelIdx, 0);
    INDArray e = org.nd4j.linalg.factory.Nd4j.base().sum(qDst.mul(kSrc.add(rel)), false, 1).mul(scale);
    INDArray alpha = org.nd4j.linalg.factory.Nd4j.sparse().csrRowSoftmax(e, rowPtr, (int) N);
    INDArray weighted = vSrc.mul(org.nd4j.linalg.factory.Nd4j.base().reshape(alpha, nnz, 1));
    INDArray out = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeAggregate(rowPtr, weighted, (int) N, 0);
    return out;
  }

  /**
   * Inner-product link decoder (Kipf & Welling 2016): Z*Z^T. Apply sigmoid downstream for edge probabilities.<br>
   *
   * @param z Node latents [N, d] (FLOATING_POINT type)
   * @return out Edge logits [N, N] (FLOATING_POINT type)
   */
  public INDArray innerProductDecoder(INDArray z) {
    NDValidation.validateFloatingPoint("innerProductDecoder", "z", z);
    INDArray out = org.nd4j.linalg.factory.Nd4j.base().mmul(z, z, false, true, false);
    return out;
  }

  /**
   * Jumping-Knowledge concatenation aggregator (Xu et al. 2018).<br>
   * Concatenates per-layer representations along the feature dimension.<br>
   *
   * @param layerOutputs Per-layer node representations, each [rows, H_l] (FLOATING_POINT type)
   * @return out Concatenated representation [rows, sum(H_l)] (FLOATING_POINT type)
   */
  public INDArray jkNetConcat(INDArray... layerOutputs) {
    NDValidation.validateFloatingPoint("jkNetConcat", "layerOutputs", layerOutputs);
    Preconditions.checkArgument(layerOutputs.length >= 1, "layerOutputs has incorrect size/length. Expected: layerOutputs.length >= 1, got %s", layerOutputs.length);
    INDArray out = org.nd4j.linalg.factory.Nd4j.base().concat(1, layerOutputs);
    return out;
  }

  /**
   * Jumping-Knowledge max-pooling aggregator (Xu et al. 2018).<br>
   * Element-wise maximum across per-layer node representations.<br>
   *
   * @param layerOutputs Per-layer node representations, each [rows, H] (FLOATING_POINT type)
   * @return out Element-wise max across layers [rows, H] (FLOATING_POINT type)
   */
  public INDArray jkNetMax(INDArray... layerOutputs) {
    NDValidation.validateFloatingPoint("jkNetMax", "layerOutputs", layerOutputs);
    Preconditions.checkArgument(layerOutputs.length >= 1, "layerOutputs has incorrect size/length. Expected: layerOutputs.length >= 1, got %s", layerOutputs.length);
    INDArray out = layerOutputs[0];
    for (int i = 1; i < layerOutputs.length; i++) {
        out = org.nd4j.linalg.factory.Nd4j.math().max(out, layerOutputs[i]);
    }
    return out;
  }

  /**
   * Edge-conditioned convolution / NNConv (Simonovsky & Komodakis 2017; Gilmer et al. MPNN 2017).<br>
   * An edge network maps edge features to [Fin, Fout] weight matrices applied to neighbour features.<br>
   *
   * @param X Node features [rows, Fin] (FLOATING_POINT type)
   * @param edgeFeatures Per-edge features [nnz, edgeF] (FLOATING_POINT type)
   * @param edgeNetW Edge-network weight [edgeF, Fin*Fout] (FLOATING_POINT type)
   * @param edgeNetB Edge-network bias [Fin*Fout] (FLOATING_POINT type)
   * @param rootW Root/self transform [Fin, Fout] (FLOATING_POINT type)
   * @param bias Optional output bias [Fout]; pass null to omit (FLOATING_POINT type)
   * @param colIdx CSR column indices [nnz] INT32 (INT type)
   * @param rowPtr CSR row pointers [rows+1] INT32 (INT type)
   * @param rows Number of nodes
   * @param n Number of nodes (for CsrEdgeGather doDiff)
   * @param fin Input feature dimension
   * @param fout Output feature dimension
   * @return out Node embeddings [rows, Fout] (FLOATING_POINT type)
   */
  public INDArray nnConv(INDArray X, INDArray edgeFeatures, INDArray edgeNetW, INDArray edgeNetB,
      INDArray rootW, INDArray bias, INDArray colIdx, INDArray rowPtr, long rows, long n, long fin,
      long fout) {
    NDValidation.validateFloatingPoint("nnConv", "X", X);
    NDValidation.validateFloatingPoint("nnConv", "edgeFeatures", edgeFeatures);
    NDValidation.validateFloatingPoint("nnConv", "edgeNetW", edgeNetW);
    NDValidation.validateFloatingPoint("nnConv", "edgeNetB", edgeNetB);
    NDValidation.validateFloatingPoint("nnConv", "rootW", rootW);
    NDValidation.validateFloatingPoint("nnConv", "bias", bias);
    NDValidation.validateInteger("nnConv", "colIdx", colIdx);
    NDValidation.validateInteger("nnConv", "rowPtr", rowPtr);
    INDArray Xj = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeGather(colIdx, X, (int) n);
    INDArray eW = org.nd4j.linalg.factory.Nd4j.base().mmul(edgeFeatures, edgeNetW).add(edgeNetB);
    INDArray eWr = org.nd4j.linalg.factory.Nd4j.base().reshape(eW, -1, fin, fout);
    INDArray XjE = org.nd4j.linalg.factory.Nd4j.base().reshape(Xj, -1, fin, 1);
    INDArray prod = XjE.mul(eWr);
    INDArray msg = org.nd4j.linalg.factory.Nd4j.base().sum(prod, false, 1);
    INDArray agg = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeAggregate(rowPtr, msg, (int) rows, 0);
    INDArray out = agg.add(org.nd4j.linalg.factory.Nd4j.base().mmul(X, rootW));
    if (bias != null) {
        out = out.add(bias);
    }
    return out;
  }

  /**
   * PairNorm (Zhao & Akoglu 2020): parameter-free normalisation that keeps total pairwise feature distance constant.<br>
   *
   * @param X Node features [rows, F] (FLOATING_POINT type)
   * @param scale Target row-norm scale s (typically ~1.0)
   * @return out Normalised node features [rows, F] (FLOATING_POINT type)
   */
  public INDArray pairNorm(INDArray X, double scale) {
    NDValidation.validateFloatingPoint("pairNorm", "X", X);
    INDArray mean = org.nd4j.linalg.factory.Nd4j.base().mean(X, true, 0);
    INDArray Xc = X.sub(mean);
    INDArray rowSq = org.nd4j.linalg.factory.Nd4j.base().sum(Xc.mul(Xc), true, 1);
    INDArray denom = org.nd4j.linalg.factory.Nd4j.math().sqrt(org.nd4j.linalg.factory.Nd4j.base().mean(rowSq, false).add(1e-6));
    INDArray out = Xc.mul(scale).div(denom);
    return out;
  }

  /**
   * Principal Neighbourhood Aggregation convolution (Corso et al. 2020).<br>
   * Combines mean, max, min and std aggregators, concatenated and linearly projected.<br>
   *
   * @param X Node features [rows, F] (FLOATING_POINT type)
   * @param W Weight matrix [4F, H] (FLOATING_POINT type)
   * @param bias Optional bias [H]; pass null to omit (FLOATING_POINT type)
   * @param colIdx CSR column indices [nnz] INT32 (INT type)
   * @param rowPtr CSR row pointers [rows+1] INT32 (INT type)
   * @param rows Number of nodes
   * @param n Number of nodes (for CsrEdgeGather doDiff)
   * @param applyRelu Whether to apply ReLU to the output
   * @return out Node embeddings [rows, H] (FLOATING_POINT type)
   */
  public INDArray pnaConv(INDArray X, INDArray W, INDArray bias, INDArray colIdx, INDArray rowPtr,
      long rows, long n, boolean applyRelu) {
    NDValidation.validateFloatingPoint("pnaConv", "X", X);
    NDValidation.validateFloatingPoint("pnaConv", "W", W);
    NDValidation.validateFloatingPoint("pnaConv", "bias", bias);
    NDValidation.validateInteger("pnaConv", "colIdx", colIdx);
    NDValidation.validateInteger("pnaConv", "rowPtr", rowPtr);
    INDArray Xj = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeGather(colIdx, X, (int) n);
    INDArray mean = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeAggregate(rowPtr, Xj, (int) rows, 1);
    INDArray max = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeAggregate(rowPtr, Xj, (int) rows, 2);
    INDArray min = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeAggregate(rowPtr, Xj.mul(-1.0), (int) rows, 2).mul(-1.0);
    INDArray meanSq = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeAggregate(rowPtr, Xj.mul(Xj), (int) rows, 1);
    INDArray std = org.nd4j.linalg.factory.Nd4j.math().sqrt(meanSq.sub(mean.mul(mean)).add(1e-6));
    INDArray agg = org.nd4j.linalg.factory.Nd4j.base().concat(1, mean, max, min, std);
    INDArray out = org.nd4j.linalg.factory.Nd4j.base().mmul(agg, W);
    if (bias != null) {
        out = out.add(bias);
    }
    if (applyRelu) {
        out = org.nd4j.linalg.factory.Nd4j.nn().relu(out, 0.0);
    }
    return out;
  }

  /**
   * Single-head relational GAT: graph attention with relation embedding modulating each edge's message.<br>
   *
   * @param X Node features [N, F] (FLOATING_POINT type)
   * @param W Linear weight [F, H] (FLOATING_POINT type)
   * @param relEmb Relation embeddings [numRelations, H] (FLOATING_POINT type)
   * @param edgeRelIdx Per-edge relation index [nnz] INT32 (INT type)
   * @param att Shared attention vector [H, 1] (FLOATING_POINT type)
   * @param colIdx CSR column indices (source nodes) [nnz] INT32 (INT type)
   * @param rowPtr CSR row pointers [N+1] INT32 (INT type)
   * @param rowIdx Per-edge destination index [nnz] INT32 (INT type)
   * @param nnz Number of edges
   * @param N Number of nodes
   * @param leakyAlpha LeakyReLU negative slope
   * @return out Aggregated node embeddings [N, H] (FLOATING_POINT type)
   */
  public INDArray rgatConvHead(INDArray X, INDArray W, INDArray relEmb, INDArray edgeRelIdx,
      INDArray att, INDArray colIdx, INDArray rowPtr, INDArray rowIdx, long nnz, long N,
      double leakyAlpha) {
    NDValidation.validateFloatingPoint("rgatConvHead", "X", X);
    NDValidation.validateFloatingPoint("rgatConvHead", "W", W);
    NDValidation.validateFloatingPoint("rgatConvHead", "relEmb", relEmb);
    NDValidation.validateInteger("rgatConvHead", "edgeRelIdx", edgeRelIdx);
    NDValidation.validateFloatingPoint("rgatConvHead", "att", att);
    NDValidation.validateInteger("rgatConvHead", "colIdx", colIdx);
    NDValidation.validateInteger("rgatConvHead", "rowPtr", rowPtr);
    NDValidation.validateInteger("rgatConvHead", "rowIdx", rowIdx);
    INDArray Wh = org.nd4j.linalg.factory.Nd4j.base().mmul(X, W);
    INDArray src = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeGather(colIdx, Wh, (int) N);
    INDArray dst = org.nd4j.linalg.factory.Nd4j.base().gather(Wh, rowIdx, 0);
    INDArray rel = org.nd4j.linalg.factory.Nd4j.base().gather(relEmb, edgeRelIdx, 0);
    INDArray msg = src.add(rel);
    INDArray g = org.nd4j.linalg.factory.Nd4j.nn().leakyRelu(msg.add(dst), leakyAlpha);
    INDArray e = org.nd4j.linalg.factory.Nd4j.base().reshape(org.nd4j.linalg.factory.Nd4j.base().mmul(g, att), nnz);
    INDArray alpha = org.nd4j.linalg.factory.Nd4j.sparse().csrRowSoftmax(e, rowPtr, (int) N);
    INDArray weighted = msg.mul(org.nd4j.linalg.factory.Nd4j.base().reshape(alpha, nnz, 1));
    INDArray out = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeAggregate(rowPtr, weighted, (int) N, 0);
    return out;
  }

  /**
   * Relational Graph Convolutional Network layer (Schlichtkrull et al. 2018).<br>
   * out = X*W_self + sum_r(A_r*X*W_r) + bias<br>
   *
   * @param X Node features [rows, F] (FLOATING_POINT type)
   * @param relVals Per-relation CSR values, relVals[r]=[nnz_r] (FLOATING_POINT type)
   * @param relColIdx Per-relation CSR column indices [nnz_r] INT32 (INT type)
   * @param relRowPtr Per-relation CSR row pointers [rows+1] INT32 (INT type)
   * @param relW Per-relation weights relW[r]=[F,H] (FLOATING_POINT type)
   * @param selfW Self-loop weight [F, H] (FLOATING_POINT type)
   * @param bias Optional bias [H]; pass null to omit (FLOATING_POINT type)
   * @param rows Number of nodes
   * @param cols Number of columns of each A_r
   * @param applyRelu Whether to apply ReLU to the output
   * @return out Node embeddings [rows, H] (FLOATING_POINT type)
   */
  public INDArray rgcnConv(INDArray X, INDArray[] relVals, INDArray[] relColIdx,
      INDArray[] relRowPtr, INDArray[] relW, INDArray selfW, INDArray bias, long rows, long cols,
      boolean applyRelu) {
    NDValidation.validateFloatingPoint("rgcnConv", "X", X);
    NDValidation.validateFloatingPoint("rgcnConv", "relVals", relVals);
    Preconditions.checkArgument(relVals.length >= 1, "relVals has incorrect size/length. Expected: relVals.length >= 1, got %s", relVals.length);
    NDValidation.validateInteger("rgcnConv", "relColIdx", relColIdx);
    Preconditions.checkArgument(relColIdx.length >= 1, "relColIdx has incorrect size/length. Expected: relColIdx.length >= 1, got %s", relColIdx.length);
    NDValidation.validateInteger("rgcnConv", "relRowPtr", relRowPtr);
    Preconditions.checkArgument(relRowPtr.length >= 1, "relRowPtr has incorrect size/length. Expected: relRowPtr.length >= 1, got %s", relRowPtr.length);
    NDValidation.validateFloatingPoint("rgcnConv", "relW", relW);
    Preconditions.checkArgument(relW.length >= 1, "relW has incorrect size/length. Expected: relW.length >= 1, got %s", relW.length);
    NDValidation.validateFloatingPoint("rgcnConv", "selfW", selfW);
    NDValidation.validateFloatingPoint("rgcnConv", "bias", bias);
    INDArray out = org.nd4j.linalg.factory.Nd4j.base().mmul(X, selfW);
    for (int r = 0; r < relW.length; r++) {
        INDArray aX = org.nd4j.linalg.factory.Nd4j.sparse().csrSpmm(relVals[r], relColIdx[r], relRowPtr[r], X, (int) rows, (int) cols, false);
        out = out.add(org.nd4j.linalg.factory.Nd4j.base().mmul(aX, relW[r]));
    }
    if (bias != null) {
        out = out.add(bias);
    }
    if (applyRelu) {
        out = org.nd4j.linalg.factory.Nd4j.nn().relu(out, 0.0);
    }
    return out;
  }

  /**
   * GraphSAGE max aggregation.<br>
   *
   * @param X Node features [rows, F] (FLOATING_POINT type)
   * @param W Weight matrix [2F, H] (FLOATING_POINT type)
   * @param bias Optional bias [H]; pass null to omit (FLOATING_POINT type)
   * @param colIdx CSR column indices [nnz] INT32 (INT type)
   * @param rowPtr CSR row pointers [rows+1] INT32 (INT type)
   * @param rows Number of nodes
   * @return out Node embeddings [rows, H] (FLOATING_POINT type)
   */
  public INDArray sageMax(INDArray X, INDArray W, INDArray bias, INDArray colIdx, INDArray rowPtr,
      long rows) {
    NDValidation.validateFloatingPoint("sageMax", "X", X);
    NDValidation.validateFloatingPoint("sageMax", "W", W);
    NDValidation.validateFloatingPoint("sageMax", "bias", bias);
    NDValidation.validateInteger("sageMax", "colIdx", colIdx);
    NDValidation.validateInteger("sageMax", "rowPtr", rowPtr);
    INDArray agg = org.nd4j.linalg.factory.Nd4j.sparse().csrSegmentMax(colIdx, rowPtr, X, (int) rows);
    INDArray cat = org.nd4j.linalg.factory.Nd4j.base().concat(1, X, agg);
    INDArray h = org.nd4j.linalg.factory.Nd4j.base().mmul(cat, W);
    if (bias != null) {
        h = h.add(bias);
    }
    INDArray out = org.nd4j.linalg.factory.Nd4j.nn().relu(h, 0.0);
    return out;
  }

  /**
   * GraphSAGE mean aggregation (Hamilton et al. 2017).<br>
   *
   * @param X Node features [rows, F] (FLOATING_POINT type)
   * @param W Weight matrix [2F, H] (FLOATING_POINT type)
   * @param bias Optional bias [H]; pass null to omit (FLOATING_POINT type)
   * @param colIdx CSR column indices [nnz] INT32 (INT type)
   * @param rowPtr CSR row pointers [rows+1] INT32 (INT type)
   * @param rows Number of nodes
   * @param n Number of nodes (= X.shape[0], for CsrEdgeGather doDiff)
   * @return out Node embeddings [rows, H] (FLOATING_POINT type)
   */
  public INDArray sageMean(INDArray X, INDArray W, INDArray bias, INDArray colIdx, INDArray rowPtr,
      long rows, long n) {
    NDValidation.validateFloatingPoint("sageMean", "X", X);
    NDValidation.validateFloatingPoint("sageMean", "W", W);
    NDValidation.validateFloatingPoint("sageMean", "bias", bias);
    NDValidation.validateInteger("sageMean", "colIdx", colIdx);
    NDValidation.validateInteger("sageMean", "rowPtr", rowPtr);
    INDArray Xj = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeGather(colIdx, X, (int) n);
    INDArray agg = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeAggregate(rowPtr, Xj, (int) rows, 1);
    INDArray cat = org.nd4j.linalg.factory.Nd4j.base().concat(1, X, agg);
    INDArray h = org.nd4j.linalg.factory.Nd4j.base().mmul(cat, W);
    if (bias != null) {
        h = h.add(bias);
    }
    INDArray out = org.nd4j.linalg.factory.Nd4j.nn().relu(h, 0.0);
    return out;
  }

  /**
   * GraphSAGE pool aggregation: apply an MLP to each neighbour, max-aggregate, then predict.<br>
   *
   * @param X Node features [rows, F] (FLOATING_POINT type)
   * @param wPool MLP weight for neighbours [F, H_pool] (FLOATING_POINT type)
   * @param bPool MLP bias [H_pool] (FLOATING_POINT type)
   * @param wOut Output weight [F + H_pool, H_out] (FLOATING_POINT type)
   * @param bOut Output bias [H_out]; pass null to omit (FLOATING_POINT type)
   * @param colIdx CSR column indices [nnz] INT32 (INT type)
   * @param rowPtr CSR row pointers [rows+1] INT32 (INT type)
   * @param rows Number of nodes
   * @param n Number of nodes (for CsrEdgeGather doDiff)
   * @return out Node embeddings [rows, H_out] (FLOATING_POINT type)
   */
  public INDArray sagePool(INDArray X, INDArray wPool, INDArray bPool, INDArray wOut, INDArray bOut,
      INDArray colIdx, INDArray rowPtr, long rows, long n) {
    NDValidation.validateFloatingPoint("sagePool", "X", X);
    NDValidation.validateFloatingPoint("sagePool", "wPool", wPool);
    NDValidation.validateFloatingPoint("sagePool", "bPool", bPool);
    NDValidation.validateFloatingPoint("sagePool", "wOut", wOut);
    NDValidation.validateFloatingPoint("sagePool", "bOut", bOut);
    NDValidation.validateInteger("sagePool", "colIdx", colIdx);
    NDValidation.validateInteger("sagePool", "rowPtr", rowPtr);
    INDArray Xj = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeGather(colIdx, X, (int) n);
    INDArray hj = org.nd4j.linalg.factory.Nd4j.nn().relu(org.nd4j.linalg.factory.Nd4j.base().mmul(Xj, wPool).add(bPool), 0.0);
    INDArray agg = org.nd4j.linalg.factory.Nd4j.sparse().csrEdgeAggregate(rowPtr, hj, (int) rows, 2);
    INDArray cat = org.nd4j.linalg.factory.Nd4j.base().concat(1, X, agg);
    INDArray out = org.nd4j.linalg.factory.Nd4j.base().mmul(cat, wOut);
    if (bOut != null) {
        out = out.add(bOut);
    }
    out = org.nd4j.linalg.factory.Nd4j.nn().relu(out, 0.0);
    return out;
  }

  /**
   * Temporal Graph Convolutional Network: weight-shared spatial GCN applied at each timestep,<br>
   * fused by temporal attention. For each timestep t: h[t] = gcnConv(X_t, W, bias, A_norm).<br>
   * Temporal attention score = mean(tanh(h[t]*tempW)*tempQ). Softmax gives beta_t;<br>
   * out = sum_t(beta_t * h[t]).<br>
   *
   * @param Xt Per-timestep node features, each [N, F] (FLOATING_POINT type)
   * @param W Shared spatial GCN weight [F, H] (FLOATING_POINT type)
   * @param bias Shared spatial GCN bias [H]; pass null to omit (FLOATING_POINT type)
   * @param aNormVals CSR values of the normalised adjacency [nnz] (FLOATING_POINT type)
   * @param aNormColIdx CSR column indices [nnz] INT32 (INT type)
   * @param aNormRowPtr CSR row pointers [rows+1] INT32 (INT type)
   * @param tempW Temporal-attention weight [H, A] (FLOATING_POINT type)
   * @param tempQ Temporal-attention query [A, 1] (FLOATING_POINT type)
   * @param rows Number of nodes
   * @param cols Columns of A_norm (= rows for square graphs)
   * @param applyRelu Whether the spatial GCN applies ReLU
   * @return out Temporally-fused node embeddings [N, H] (FLOATING_POINT type)
   */
  public INDArray temporalGcn(INDArray[] Xt, INDArray W, INDArray bias, INDArray aNormVals,
      INDArray aNormColIdx, INDArray aNormRowPtr, INDArray tempW, INDArray tempQ, long rows,
      long cols, boolean applyRelu) {
    NDValidation.validateFloatingPoint("temporalGcn", "Xt", Xt);
    Preconditions.checkArgument(Xt.length >= 1, "Xt has incorrect size/length. Expected: Xt.length >= 1, got %s", Xt.length);
    NDValidation.validateFloatingPoint("temporalGcn", "W", W);
    NDValidation.validateFloatingPoint("temporalGcn", "bias", bias);
    NDValidation.validateFloatingPoint("temporalGcn", "aNormVals", aNormVals);
    NDValidation.validateInteger("temporalGcn", "aNormColIdx", aNormColIdx);
    NDValidation.validateInteger("temporalGcn", "aNormRowPtr", aNormRowPtr);
    NDValidation.validateFloatingPoint("temporalGcn", "tempW", tempW);
    NDValidation.validateFloatingPoint("temporalGcn", "tempQ", tempQ);
    final int T = Xt.length;
    INDArray[] h = new INDArray[T];
    INDArray[] expw = new INDArray[T];
    INDArray sumExp = null;
    for (int t = 0; t < T; t++) {
        h[t] = org.nd4j.linalg.factory.Nd4j.gnn().gcnConv(Xt[t], W, bias, aNormVals, aNormColIdx, aNormRowPtr, rows, cols, applyRelu);
        INDArray score = org.nd4j.linalg.factory.Nd4j.base().mean(org.nd4j.linalg.factory.Nd4j.base().mmul(org.nd4j.linalg.factory.Nd4j.math().tanh(org.nd4j.linalg.factory.Nd4j.base().mmul(h[t], tempW)), tempQ), false);
        expw[t] = org.nd4j.linalg.factory.Nd4j.math().exp(score);
        sumExp = (t == 0) ? expw[t] : sumExp.add(expw[t]);
    }
    INDArray out = null;
    for (int t = 0; t < T; t++) {
        INDArray betaT = expw[t].div(sumExp);
        INDArray term  = h[t].mul(betaT);
        out = (t == 0) ? term : out.add(term);
    }
    return out;
  }

  /**
   * VGAE KL-divergence regulariser (Kipf & Welling 2016): 0.5 * mean(exp(logvar) + mu^2 - logvar - 1)<br>
   *
   * @param mu Latent mean [N, d] (FLOATING_POINT type)
   * @param logvar Latent log-variance [N, d] (FLOATING_POINT type)
   * @return out Scalar KL divergence (FLOATING_POINT type)
   */
  public INDArray vgaeKlLoss(INDArray mu, INDArray logvar) {
    NDValidation.validateFloatingPoint("vgaeKlLoss", "mu", mu);
    NDValidation.validateFloatingPoint("vgaeKlLoss", "logvar", logvar);
    INDArray kl = org.nd4j.linalg.factory.Nd4j.math().exp(logvar).add(mu.mul(mu)).sub(logvar).sub(1.0);
    INDArray out = org.nd4j.linalg.factory.Nd4j.base().mean(kl, false).mul(0.5);
    return out;
  }

  /**
   * VGAE reparameterisation trick (Kipf & Welling 2016): z = mu + exp(0.5*logvar) * noise<br>
   *
   * @param mu Latent mean [N, d] (FLOATING_POINT type)
   * @param logvar Latent log-variance [N, d] (FLOATING_POINT type)
   * @param noise Standard-normal sample [N, d] (FLOATING_POINT type)
   * @return out Latent sample z [N, d] (FLOATING_POINT type)
   */
  public INDArray vgaeReparam(INDArray mu, INDArray logvar, INDArray noise) {
    NDValidation.validateFloatingPoint("vgaeReparam", "mu", mu);
    NDValidation.validateFloatingPoint("vgaeReparam", "logvar", logvar);
    NDValidation.validateFloatingPoint("vgaeReparam", "noise", noise);
    INDArray out = mu.add(org.nd4j.linalg.factory.Nd4j.math().exp(logvar.mul(0.5)).mul(noise));
    return out;
  }
}
