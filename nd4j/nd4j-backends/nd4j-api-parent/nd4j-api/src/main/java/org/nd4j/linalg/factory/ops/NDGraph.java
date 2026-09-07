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
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.sparse.GraphDisjointUnion;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.factory.Nd4j;

public class NDGraph {
  public NDGraph() {
  }

  /**
   * Adamic-Adar link-prediction score: S[i,j] = sum_v A[i,v]·A[v,j] / log(deg_v), weighting each
   * shared neighbor v by the inverse log of its degree so that rare (low-degree) common neighbors
   * contribute more than hubs. Assumes node degrees > 1 (so log(deg) > 0).
   *
   * @param adj Adjacency matrix [n, n] with node degrees > 1 (FLOATING_POINT type)
   * @return score Adamic-Adar score matrix [n, n] (FLOATING_POINT type)
   */
  public INDArray adamicAdar(INDArray adj) {
    NDValidation.validateFloatingPoint("adamicAdar", "adj", adj);
    INDArray deg = Nd4j.base().sum(adj, false, 1);
    INDArray logDeg = Nd4j.math().log(deg);
    INDArray scaled = adj.div(Nd4j.base().reshape(logDeg, -1, 1));
    INDArray out = Nd4j.base().mmul(adj, scaled);
    return out;
  }

  /**
   * BGRL (Bootstrap Your Own Latent for Graphs, Thakoor et al. 2022) self-supervised loss.
   * Computes the row-normalized cosine similarity between the online prediction (onlineZ @ predW)
   * and the target embedding (targetZ), then takes the mean of (2 - 2·cosine). The minimum loss
   * is 0 (perfect alignment); gradients flow only through onlineZ and predW -- declare targetZ as
   * sd.constant(...) in the calling code to implement the stop-gradient.
   * loss = mean_i( 2 - 2 * cosine( (onlineZ @ predW)_i, targetZ_i ) )
   *
   * @param onlineZ Online encoder node embeddings [n, d] (FLOATING_POINT type)
   * @param targetZ Target (stop-grad) node embeddings [n, d] (FLOATING_POINT type)
   * @param predW Online predictor weight matrix [d, d] (FLOATING_POINT type)
   * @return loss Scalar BGRL loss: mean(2 - 2·cosine(onlineZ·predW, targetZ)) (FLOATING_POINT type)
   */
  public INDArray bgrlLoss(INDArray onlineZ, INDArray targetZ, INDArray predW) {
    NDValidation.validateFloatingPoint("bgrlLoss", "onlineZ", onlineZ);
    NDValidation.validateFloatingPoint("bgrlLoss", "targetZ", targetZ);
    NDValidation.validateFloatingPoint("bgrlLoss", "predW", predW);
    INDArray pred = Nd4j.base().mmul(onlineZ, predW);
    INDArray predNorm = pred.div(Nd4j.math().sqrt(Nd4j.base().sum(pred.mul(pred), true, 1).add(1e-12)));
    INDArray tNorm = targetZ.div(Nd4j.math().sqrt(Nd4j.base().sum(targetZ.mul(targetZ), true, 1).add(1e-12)));
    INDArray cos = Nd4j.base().sum(predNorm.mul(tNorm), false, 1);
    INDArray out = Nd4j.base().mean(cos.mul(-2.0).add(2.0));
    return out;
  }

  /**
   * Local clustering coefficient per node: C_i = (closed triangles through i) / (possible pairs) =
   * (A^3)_ii / (deg_i (deg_i - 1)). (A^3)_ii counts the length-3 closed walks through node i (= twice
   * its triangle count), computed as diag(A·A·A) via an identity-mask extraction. Measures how
   * tightly each node's neighborhood is interconnected. Pass identity = sd.constant(Nd4j.eye(n)).
   *
   * @param adj Adjacency matrix [n, n] (symmetric for undirected graphs) (FLOATING_POINT type)
   * @param identity Identity matrix [n, n] -- pass sd.constant(Nd4j.eye(n)) (FLOATING_POINT type)
   * @return coeff Per-node local clustering coefficient [n] (FLOATING_POINT type)
   */
  public INDArray clusteringCoefficient(INDArray adj, INDArray identity) {
    NDValidation.validateFloatingPoint("clusteringCoefficient", "adj", adj);
    NDValidation.validateFloatingPoint("clusteringCoefficient", "identity", identity);
    INDArray a2 = Nd4j.base().mmul(adj, adj);
    INDArray a3 = Nd4j.base().mmul(a2, adj);
    INDArray tri = Nd4j.base().sum(a3.mul(identity), false, 1);
    INDArray deg = Nd4j.base().sum(adj, false, 1);
    INDArray denom = deg.mul(deg.sub(1.0)).add(1e-9);
    INDArray out = tri.div(denom);
    return out;
  }

  /**
   * Common-Neighbors link-prediction score: S = A·A, so S[i,j] counts the (weighted) paths of
   * length two between i and j -- the number of neighbors they share. The simplest topological
   * link predictor; higher scores indicate more likely missing edges.
   *
   * @param adj Adjacency matrix [n, n] (symmetric for undirected graphs) (FLOATING_POINT type)
   * @return score Common-neighbor score matrix [n, n] (FLOATING_POINT type)
   */
  public INDArray commonNeighbors(INDArray adj) {
    NDValidation.validateFloatingPoint("commonNeighbors", "adj", adj);
    INDArray out = Nd4j.base().mmul(adj, adj);
    return out;
  }

  /**
   * ComplEx (Trouillon et al. 2016): a complex-valued bilinear product whose real part scores the
   * triple; the imaginary parts let it model asymmetric relations.
   * score = Re( sum_d head_d * relation_d * conj(tail_d) )
   *
   * @param hRe Real part of head embeddings [batch, dim] (FLOATING_POINT type)
   * @param hIm Imag part of head embeddings [batch, dim] (FLOATING_POINT type)
   * @param rRe Real part of relation embeddings [batch, dim] (FLOATING_POINT type)
   * @param rIm Imag part of relation embeddings [batch, dim] (FLOATING_POINT type)
   * @param tRe Real part of tail embeddings [batch, dim] (FLOATING_POINT type)
   * @param tIm Imag part of tail embeddings [batch, dim] (FLOATING_POINT type)
   * @return score Plausibility score [batch] (higher = better) (FLOATING_POINT type)
   */
  public INDArray complEx(INDArray hRe, INDArray hIm, INDArray rRe, INDArray rIm, INDArray tRe,
      INDArray tIm) {
    NDValidation.validateFloatingPoint("complEx", "hRe", hRe);
    NDValidation.validateFloatingPoint("complEx", "hIm", hIm);
    NDValidation.validateFloatingPoint("complEx", "rRe", rRe);
    NDValidation.validateFloatingPoint("complEx", "rIm", rIm);
    NDValidation.validateFloatingPoint("complEx", "tRe", tRe);
    NDValidation.validateFloatingPoint("complEx", "tIm", tIm);
    INDArray s = hRe.mul(rRe).mul(tRe).add(hRe.mul(rIm).mul(tIm)).add(hIm.mul(rRe).mul(tIm)).sub(hIm.mul(rIm).mul(tRe));
    INDArray out = Nd4j.base().sum(s, false, 1);
    return out;
  }

  /**
   * ConvE (Dettmers et al. 2018): reshape head + relation into 2D images, stack, run a 2D
   * convolution + fully-connected projection, then score against the tail. A strong,
   * parameter-efficient KGE baseline.
   *
   * @param head Head-entity embeddings [batch, de] (de = embH*embW) (FLOATING_POINT type)
   * @param relation Relation embeddings [batch, de] (FLOATING_POINT type)
   * @param tail Tail-entity embeddings [batch, de] (FLOATING_POINT type)
   * @param convW Conv weights [3, 3, 1, channels] (FLOATING_POINT type)
   * @param convB Conv bias [channels] (FLOATING_POINT type)
   * @param fcW FC weights [channels*(2*embH-2)*(embW-2), de] (FLOATING_POINT type)
   * @param fcB FC bias [de] (FLOATING_POINT type)
   * @param embH Reshape height per embedding
   * @param embW Reshape width (de = embH*embW)
   * @param channels Conv output channels
   * @return score Plausibility score [batch] (higher = better) (FLOATING_POINT type)
   */
  public INDArray convE(INDArray head, INDArray relation, INDArray tail, INDArray convW,
      INDArray convB, INDArray fcW, INDArray fcB, int embH, int embW, int channels) {
    NDValidation.validateFloatingPoint("convE", "head", head);
    NDValidation.validateFloatingPoint("convE", "relation", relation);
    NDValidation.validateFloatingPoint("convE", "tail", tail);
    NDValidation.validateFloatingPoint("convE", "convW", convW);
    NDValidation.validateFloatingPoint("convE", "convB", convB);
    NDValidation.validateFloatingPoint("convE", "fcW", fcW);
    NDValidation.validateFloatingPoint("convE", "fcB", fcB);
    INDArray hImg = Nd4j.base().reshape(head, -1, embH, embW);
    INDArray rImg = Nd4j.base().reshape(relation, -1, embH, embW);
    INDArray stacked = Nd4j.base().concat(1, hImg, rImg);
    INDArray img = Nd4j.base().reshape(stacked, -1, 1, 2L * embH, embW);
    Conv2DConfig cfg = Conv2DConfig.builder().kH(3).kW(3).build();
    INDArray feat = Nd4j.nn().relu(Nd4j.cnn().conv2d(img, convW, convB, cfg), 0.0);
    long flatDim = (long) channels * (2L * embH - 2) * (embW - 2);
    INDArray flat = Nd4j.base().reshape(feat, -1, flatDim);
    INDArray fc = Nd4j.nn().relu(Nd4j.base().mmul(flat, fcW).add(fcB), 0.0);
    INDArray out = Nd4j.base().sum(fc.mul(tail), false, 1);
    return out;
  }

  /**
   * Correct &amp; Smooth post-processing (Huang et al. 2020): two-phase label diffusion over a graph.
   * Phase 1 (Correct, iter1 steps): spreads label residuals E over the graph while retaining a
   * fraction (1-alpha1) of the original residuals at every step, then adds the spread residuals to
   * the base predictions. Phase 2 (Smooth, iter2 steps): diffuses the corrected predictions while
   * retaining fraction (1-alpha2) of the corrected values. Both phases use the same normalized
   * adjacency A_norm.
   *
   * @param basePreds Base per-node predictions / logits [n, c] (FLOATING_POINT type)
   * @param aNorm Dense (row- or sym-) normalized adjacency [n, n] (FLOATING_POINT type)
   * @param residuals Label residuals at training nodes, zeros elsewhere [n, c] (FLOATING_POINT type)
   * @param alpha1 Residual-spreading weight in (0,1) for the Correct phase
   * @param alpha2 Label-smoothing weight in (0,1) for the Smooth phase
   * @param iter1 Number of Correct-phase propagation steps
   * @param iter2 Number of Smooth-phase propagation steps
   * @return out Corrected and smoothed per-node predictions [n, c] (FLOATING_POINT type)
   */
  public INDArray correctAndSmooth(INDArray basePreds, INDArray aNorm, INDArray residuals,
      double alpha1, double alpha2, int iter1, int iter2) {
    NDValidation.validateFloatingPoint("correctAndSmooth", "basePreds", basePreds);
    NDValidation.validateFloatingPoint("correctAndSmooth", "aNorm", aNorm);
    NDValidation.validateFloatingPoint("correctAndSmooth", "residuals", residuals);
    INDArray E = residuals;
    for (int i = 0; i < iter1; i++) {
        E = Nd4j.base().mmul(aNorm, E).mul(alpha1).add(residuals.mul(1.0 - alpha1));
    }
    INDArray corrected = basePreds.add(E);
    INDArray Y = corrected;
    for (int j = 0; j < iter2; j++) {
        Y = Nd4j.base().mmul(aNorm, Y).mul(alpha2).add(corrected.mul(1.0 - alpha2));
    }
    INDArray out = Y;
    return out;
  }

  /**
   * Pearson correlation matrix among the columns (variables) of a data matrix -- the basis of a
   * correlation graph. Threshold |corr| with sd.sparse().denseToCsr(...) to obtain a sparse
   * correlation-graph adjacency.
   *
   * @param data Observations x variables [n, d] (FLOATING_POINT type)
   * @return corr Pearson correlation matrix [d, d] with unit diagonal (FLOATING_POINT type)
   */
  public INDArray correlationMatrix(INDArray data) {
    NDValidation.validateFloatingPoint("correlationMatrix", "data", data);
    INDArray mean = Nd4j.base().mean(data, true, 0);
    INDArray centered = data.sub(mean);
    INDArray cov = Nd4j.base().mmul(centered, centered, true, false, false);
    INDArray variance = Nd4j.base().sum(centered.mul(centered), false, 0);
    INDArray std = Nd4j.math().sqrt(variance.add(1e-12));
    INDArray denom = Nd4j.base().mmul(Nd4j.base().reshape(std, -1, 1), Nd4j.base().reshape(std, 1, -1));
    INDArray out = cov.div(denom);
    return out;
  }

  /**
   * Cosine-similarity affinity between the rows (nodes) of a feature matrix.
   * sim[i,j] = (x_i . x_j) / (||x_i|| ||x_j||). A standard input to a kNN / thresholded similarity graph.
   *
   * @param features Node features [n, d] (FLOATING_POINT type)
   * @return sim Cosine-similarity matrix [n, n] with unit diagonal (FLOATING_POINT type)
   */
  public INDArray cosineSimilarity(INDArray features) {
    NDValidation.validateFloatingPoint("cosineSimilarity", "features", features);
    INDArray norm = Nd4j.math().sqrt(Nd4j.base().sum(features.mul(features), true, 1).add(1e-12));
    INDArray normalized = features.div(norm);
    INDArray out = Nd4j.base().mmul(normalized, normalized, false, true, false);
    return out;
  }

  /**
   * Deep Graph Infomax loss (Velickovic et al. 2019): maximizes mutual information between each
   * node's encoding and a global graph summary via a bilinear discriminator that tells real node
   * encodings from encodings of a corrupted graph. Yields label-free node embeddings.
   *
   * @param H Node encodings of the real graph [n, d] (FLOATING_POINT type)
   * @param Hneg Node encodings of the corrupted graph [n, d] (FLOATING_POINT type)
   * @param discW Bilinear discriminator weight [d, d] (FLOATING_POINT type)
   * @return loss Scalar DGI loss (FLOATING_POINT type)
   */
  public INDArray dgiLoss(INDArray H, INDArray Hneg, INDArray discW) {
    NDValidation.validateFloatingPoint("dgiLoss", "H", H);
    NDValidation.validateFloatingPoint("dgiLoss", "Hneg", Hneg);
    NDValidation.validateFloatingPoint("dgiLoss", "discW", discW);
    INDArray s = Nd4j.nn().sigmoid(Nd4j.base().mean(H, true, 0));
    INDArray sW = Nd4j.base().mmul(s, discW, false, true, false);
    INDArray pos = Nd4j.base().sum(H.mul(sW), false, 1);
    INDArray neg = Nd4j.base().sum(Hneg.mul(sW), false, 1);
    INDArray posTerm = Nd4j.math().log(Nd4j.nn().sigmoid(pos).add(1e-12));
    INDArray negTerm = Nd4j.math().log(Nd4j.nn().sigmoid(neg.mul(-1.0)).add(1e-12));
    INDArray out = Nd4j.base().mean(posTerm.add(negTerm), false).mul(-1.0);
    return out;
  }

  /**
   * DistMult (Yang et al. 2015): a symmetric trilinear product.
   * score = sum_d head_d * relation_d * tail_d
   *
   * @param head Head-entity embeddings [batch, dim] (FLOATING_POINT type)
   * @param relation Relation embeddings [batch, dim] (FLOATING_POINT type)
   * @param tail Tail-entity embeddings [batch, dim] (FLOATING_POINT type)
   * @return score Plausibility score [batch] (higher = better) (FLOATING_POINT type)
   */
  public INDArray distMult(INDArray head, INDArray relation, INDArray tail) {
    NDValidation.validateFloatingPoint("distMult", "head", head);
    NDValidation.validateFloatingPoint("distMult", "relation", relation);
    NDValidation.validateFloatingPoint("distMult", "tail", tail);
    INDArray out = Nd4j.base().sum(head.mul(relation).mul(tail), false, 1);
    return out;
  }

  /**
   * Gaussian (RBF) similarity affinity between the rows (nodes) of a feature matrix.
   * sim[i,j] = exp( -||x_i - x_j||^2 / (2 sigma^2) ). Threshold or take per-row top-k for a kNN graph.
   *
   * @param features Node features [n, d] (FLOATING_POINT type)
   * @param sigma RBF kernel bandwidth (> 0)
   * @return sim Gaussian-similarity matrix [n, n] with unit diagonal (FLOATING_POINT type)
   */
  public INDArray gaussianSimilarity(INDArray features, double sigma) {
    NDValidation.validateFloatingPoint("gaussianSimilarity", "features", features);
    INDArray xi = Nd4j.base().expandDims(features, 1);
    INDArray xj = Nd4j.base().expandDims(features, 0);
    INDArray diff = xi.sub(xj);
    INDArray d2 = Nd4j.base().sum(diff.mul(diff), false, 2);
    INDArray out = Nd4j.math().exp(d2.mul(-1.0 / (2.0 * sigma * sigma)));
    return out;
  }

  /**
   * GRACE graph-contrastive loss (Zhu et al. 2020): an InfoNCE objective over two augmented views.
   * For each node i, the positive pair is (z1_i, z2_i) and the negatives are all z2_j (j != i):
   * loss = -mean( cosine(z1_i,z2_i)/tau - logsumexp_j cosine(z1_i,z2_j)/tau ).
   * Yields label-free node embeddings. Pass identity = sd.constant(Nd4j.eye(n).castTo(DataType.DOUBLE))
   * so the diagonal (positive) similarities are extracted with a gradient-clean elementwise mask.
   *
   * @param z1 Node embeddings of augmented view 1 [n, d] (FLOATING_POINT type)
   * @param z2 Node embeddings of augmented view 2 [n, d] (FLOATING_POINT type)
   * @param identity Identity matrix [n, n] -- pass sd.constant(Nd4j.eye(n)) (FLOATING_POINT type)
   * @param tau Temperature (e.g. 0.5)
   * @return loss Scalar GRACE / InfoNCE contrastive loss (FLOATING_POINT type)
   */
  public INDArray graceLoss(INDArray z1, INDArray z2, INDArray identity, double tau) {
    NDValidation.validateFloatingPoint("graceLoss", "z1", z1);
    NDValidation.validateFloatingPoint("graceLoss", "z2", z2);
    NDValidation.validateFloatingPoint("graceLoss", "identity", identity);
    INDArray z1n = z1.div(Nd4j.math().sqrt(Nd4j.base().sum(z1.mul(z1), true, 1).add(1e-12)));
    INDArray z2n = z2.div(Nd4j.math().sqrt(Nd4j.base().sum(z2.mul(z2), true, 1).add(1e-12)));
    INDArray sim = Nd4j.base().mmul(z1n, z2n, false, true, false).div(tau);
    INDArray pos = Nd4j.base().sum(sim.mul(identity), false, 1);
    INDArray lse = Nd4j.math().log(Nd4j.base().sum(Nd4j.math().exp(sim), false, 1).add(1e-12));
    INDArray out = Nd4j.base().mean(lse.sub(pos));
    return out;
  }

  /**
   * Assembles K variable-size graphs into one block-diagonal graph for batched message passing.
   *
   * The resulting block-diagonal CSR is compatible with all sd.gnn() message-passing ops.
   * Use batchVec with sd.segmentMean/Sum/Max for graph-level readout.
   *
   * @param Xs K node-feature matrices [N_k, F]; K is inferred from the array length (FLOATING_POINT type)
   * @param vals K edge-weight arrays [nnz_k] (FLOATING_POINT type)
   * @param colIdxs K column-index arrays [nnz_k] (INT type)
   * @param rowPtrs K row-pointer arrays [N_k+1] (INT type)
   * @return Xcombined Combined node features [sumN, F] (FLOATING_POINT type)
   * @return valsCombined Combined edge weights [sumNnz] (FLOATING_POINT type)
   * @return colIdxCombined Combined shifted column indices [sumNnz] (INT type)
   * @return rowPtrCombined Combined stitched row pointers [sumN+1] (INT type)
   * @return batchVec Node-to-graph assignment [sumN] (INT type)
   */
  public INDArray[] graphDisjointUnion(INDArray[] Xs, INDArray[] vals, INDArray[] colIdxs,
      INDArray... rowPtrs) {
    NDValidation.validateFloatingPoint("graphDisjointUnion", "Xs", Xs);
    Preconditions.checkArgument(Xs.length >= 1, "Xs has incorrect size/length. Expected: Xs.length >= 1, got %s", Xs.length);
    NDValidation.validateFloatingPoint("graphDisjointUnion", "vals", vals);
    Preconditions.checkArgument(vals.length >= 1, "vals has incorrect size/length. Expected: vals.length >= 1, got %s", vals.length);
    NDValidation.validateInteger("graphDisjointUnion", "colIdxs", colIdxs);
    Preconditions.checkArgument(colIdxs.length >= 1, "colIdxs has incorrect size/length. Expected: colIdxs.length >= 1, got %s", colIdxs.length);
    NDValidation.validateInteger("graphDisjointUnion", "rowPtrs", rowPtrs);
    Preconditions.checkArgument(rowPtrs.length >= 1, "rowPtrs has incorrect size/length. Expected: rowPtrs.length >= 1, got %s", rowPtrs.length);
    return Nd4j.exec(new GraphDisjointUnion(Xs, vals, colIdxs, rowPtrs));
  }

  /**
   * HolE -- Holographic Embeddings (Nickel et al. 2016): scores a triple by the relation's
   * agreement with the circular correlation of head and tail,
   * score = relation . ccorr(head, tail), where ccorr(a,b) = IDFT(conj(DFT(a)) * DFT(b)).
   * Circular correlation gives ComplEx-level expressiveness (asymmetric relations) at O(d log d)
   * via the Fourier domain; here it is composed from the differentiable DFT op.
   *
   * @param head Head entity embeddings [batch, dim] (FLOATING_POINT type)
   * @param relation Relation embeddings [batch, dim] (FLOATING_POINT type)
   * @param tail Tail entity embeddings [batch, dim] (FLOATING_POINT type)
   * @return score Plausibility score [batch] (higher = better) (FLOATING_POINT type)
   */
  public INDArray holE(INDArray head, INDArray relation, INDArray tail) {
    NDValidation.validateFloatingPoint("holE", "head", head);
    NDValidation.validateFloatingPoint("holE", "relation", relation);
    NDValidation.validateFloatingPoint("holE", "tail", tail);
    INDArray hExp = Nd4j.base().expandDims(head, 2);
    INDArray one = Nd4j.base().onesLike(hExp);
    INDArray zero = Nd4j.base().zerosLike(hExp);
    INDArray reSel = Nd4j.base().concat(2, one, zero);
    INDArray imSel = Nd4j.base().concat(2, zero, one);
    INDArray hc = Nd4j.base().concat(2, hExp, zero);
    INDArray tc = Nd4j.base().concat(2, Nd4j.base().expandDims(tail, 2), zero);
    INDArray H = Nd4j.signal().dft(hc, 1, false, false);
    INDArray T = Nd4j.signal().dft(tc, 1, false, false);
    INDArray hRe = Nd4j.base().sum(H.mul(reSel), false, 2);
    INDArray hIm = Nd4j.base().sum(H.mul(imSel), false, 2);
    INDArray tRe = Nd4j.base().sum(T.mul(reSel), false, 2);
    INDArray tIm = Nd4j.base().sum(T.mul(imSel), false, 2);
    INDArray zRe = hRe.mul(tRe).add(hIm.mul(tIm));
    INDArray zIm = hRe.mul(tIm).sub(hIm.mul(tRe));
    INDArray zc = Nd4j.base().concat(2, Nd4j.base().expandDims(zRe, 2), Nd4j.base().expandDims(zIm, 2));
    INDArray ccorr = Nd4j.signal().dft(zc, 1, true, false);
    INDArray ccorrRe = Nd4j.base().sum(ccorr.mul(reSel), false, 2);
    INDArray out = Nd4j.base().sum(relation.mul(ccorrRe), false, 1);
    return out;
  }

  /**
   * Topological Jaccard link-prediction score: S[i,j] = |N(i) ∩ N(j)| / |N(i) ∪ N(j)|, computed
   * as commonNeighbors(i,j) / (deg_i + deg_j - commonNeighbors(i,j)). Unlike feature-vector
   * Jaccard distance, this measures overlap of graph neighborhoods, normalizing for node degree.
   *
   * @param adj Adjacency matrix [n, n] (symmetric for undirected graphs) (FLOATING_POINT type)
   * @return score Topological Jaccard score matrix [n, n] (FLOATING_POINT type)
   */
  public INDArray jaccardTopology(INDArray adj) {
    NDValidation.validateFloatingPoint("jaccardTopology", "adj", adj);
    INDArray cn = Nd4j.base().mmul(adj, adj);
    INDArray deg = Nd4j.base().sum(adj, false, 1);
    INDArray ones = Nd4j.base().onesLike(deg);
    INDArray degI = Nd4j.base().mmul(Nd4j.base().reshape(deg, -1, 1), Nd4j.base().reshape(ones, 1, -1));
    INDArray degJ = Nd4j.base().mmul(Nd4j.base().reshape(ones, -1, 1), Nd4j.base().reshape(deg, 1, -1));
    INDArray union = degI.add(degJ).sub(cn);
    INDArray out = cn.div(union.add(1e-9));
    return out;
  }

  /**
   * Katz similarity index (Katz 1953): a link-prediction / node-similarity score that counts all
   * paths between pairs of nodes, exponentially down-weighted by path length.
   * S = sum_{l=1}^{L} beta^l A^l  (truncated finite-sum approximation).
   * Unlike the closed-form (I - beta*A)^{-1} - I this formulation uses only matrix multiplication
   * and is fully differentiable via standard mmul backward. Requires 0 &lt; beta &lt; 1/spectral_radius(A)
   * for the series to be meaningful; in practice beta = 0.05..0.1 and L = 3..5 works well.
   *
   * @param adj Adjacency matrix [n, n] (FLOATING_POINT type)
   * @param beta Attenuation factor (0 &lt; beta &lt; 1 / spectral_radius(A))
   * @param L Truncation depth: number of path-length terms (>= 1)
   * @return out Katz similarity matrix [n, n] (FLOATING_POINT type)
   */
  public INDArray katzIndex(INDArray adj, double beta, int L) {
    NDValidation.validateFloatingPoint("katzIndex", "adj", adj);
    INDArray S = adj.mul(beta);
    INDArray Apow = adj;
    double betaPow = beta;
    for (int l = 2; l <= L; l++) {
        Apow = Nd4j.base().mmul(Apow, adj);
        betaPow = betaPow * beta;
        S = S.add(Apow.mul(betaPow));
    }
    INDArray out = S;
    return out;
  }

  /**
   * k-nearest-neighbor graph construction from a similarity/affinity matrix: each node (row) keeps
   * only its k highest-similarity neighbors, zeroing the rest, yielding a sparse weighted adjacency.
   * Built by scattering the per-row top-k values back to their column positions
   * (sum over k of oneHot(topIndices, n) * topValues), so it is differentiable w.r.t. the kept
   * similarities through the TopK gradient. Pair with sd.graph().cosineSimilarity / gaussianSimilarity
   * / correlationMatrix to go from raw features straight to a learnable kNN graph. `n` is the node
   * count (matrix dimension), used as the one-hot depth.
   *
   * @param similarity Pairwise similarity / affinity matrix [n, n] (FLOATING_POINT type)
   * @param k Number of nearest neighbors to keep per row
   * @param n Number of nodes (matrix dimension); the one-hot depth
   * @return adj kNN adjacency [n, n]: each row keeps its top-k similarities, rest 0 (FLOATING_POINT type)
   */
  public INDArray knnGraph(INDArray similarity, int k, int n) {
    NDValidation.validateFloatingPoint("knnGraph", "similarity", similarity);
    INDArray[] tk = Nd4j.nn().topK(similarity, k, false);
    INDArray oneHotIdx = Nd4j.base().oneHot(tk[1], n, -1, 1.0, 0.0, similarity.dataType());
    INDArray valsExp = Nd4j.base().expandDims(tk[0], -1);
    INDArray out = Nd4j.base().sum(oneHotIdx.mul(valsExp), false, 1);
    return out;
  }

  /**
   * Personalized-PageRank label propagation (Zhou et al. 2004; APPNP propagation of labels).
   * Diffuses seed/observed label rows over the graph while retaining a fraction alpha of the seed
   * at every step: Y = (1-alpha)*(A_norm . Y) + alpha*seedY. A transductive (semi-supervised) classifier.
   *
   * @param seedY Seed label distribution per node [n, c] (FLOATING_POINT type)
   * @param aNormVals CSR values of the normalized adjacency [nnz] (FLOATING_POINT type)
   * @param aNormColIdx CSR column indices [nnz] (INT32) (INT type)
   * @param aNormRowPtr CSR row pointers [n+1] (INT32) (INT type)
   * @param rows Number of nodes
   * @param cols Columns of A_norm (= rows for square graphs)
   * @param k Number of propagation steps
   * @param alpha Teleport / restart probability in [0,1]
   * @return labels Propagated label distribution [n, c] (FLOATING_POINT type)
   */
  public INDArray labelPropagation(INDArray seedY, INDArray aNormVals, INDArray aNormColIdx,
      INDArray aNormRowPtr, int rows, int cols, int k, double alpha) {
    NDValidation.validateFloatingPoint("labelPropagation", "seedY", seedY);
    NDValidation.validateFloatingPoint("labelPropagation", "aNormVals", aNormVals);
    NDValidation.validateInteger("labelPropagation", "aNormColIdx", aNormColIdx);
    NDValidation.validateInteger("labelPropagation", "aNormRowPtr", aNormRowPtr);
    INDArray Y = seedY;
    for (int i = 0; i < k; i++) {
        INDArray AY = Nd4j.sparse().csrSpmm(aNormVals, aNormColIdx, aNormRowPtr, Y, rows, cols, false);
        Y = AY.mul(1.0 - alpha).add(seedY.mul(alpha));
    }
    INDArray out = Y;
    return out;
  }

  /**
   * Margin ranking loss for KGE training: pushes positive triples to score higher than negatives
   * by at least margin.
   * loss = mean( max(0, margin - posScore + negScore) )
   *
   * @param posScore Scores of true triples [batch] (FLOATING_POINT type)
   * @param negScore Scores of corrupted (negative) triples [batch] (FLOATING_POINT type)
   * @param margin Desired score margin
   * @return loss Scalar margin ranking loss (FLOATING_POINT type)
   */
  public INDArray marginRankingLoss(INDArray posScore, INDArray negScore, double margin) {
    NDValidation.validateFloatingPoint("marginRankingLoss", "posScore", posScore);
    NDValidation.validateFloatingPoint("marginRankingLoss", "negScore", negScore);
    INDArray hinge = Nd4j.nn().relu(negScore.sub(posScore).add(margin), 0.0);
    INDArray out = Nd4j.base().mean(hinge);
    return out;
  }

  /**
   * Personalized PageRank (PPR) via power iteration: r_{t+1} = alpha * A_norm * r_t + (1-alpha) * seed.
   * At each step the walker follows graph edges (alpha) or teleports back to the personalization seed
   * (1-alpha). After `iterations` steps the result approximates (I - alpha*A_norm)^{-1}*(1-alpha)*seed
   * without ever forming a matrix inverse, making it fully differentiable through mmul and scalar ops.
   * Use aNorm = row-normalized adjacency (each row sums to 1). For node-classification, seed = one-hot
   * class indicators [n,c]; for link prediction, seed = one-hot per-query-node vectors [n,n].
   *
   * @param aNorm Row-normalized adjacency [n, n] (FLOATING_POINT type)
   * @param seed Personalization / seed distribution [n, c] (FLOATING_POINT type)
   * @param alpha Propagation weight (teleport = 1-alpha) in (0,1)
   * @param iterations Number of power iterations
   * @return out Personalized PageRank score matrix [n, c] (FLOATING_POINT type)
   */
  public INDArray personalizedPageRank(INDArray aNorm, INDArray seed, double alpha,
      int iterations) {
    NDValidation.validateFloatingPoint("personalizedPageRank", "aNorm", aNorm);
    NDValidation.validateFloatingPoint("personalizedPageRank", "seed", seed);
    INDArray r = seed;
    for (int i = 0; i < iterations; i++) {
        r = Nd4j.base().mmul(aNorm, r).mul(alpha).add(seed.mul(1.0 - alpha));
    }
    INDArray out = r;
    return out;
  }

  /**
   * Preferential-Attachment link-prediction score: S[i,j] = deg_i · deg_j (the outer product of
   * the degree vector). Encodes the "rich get richer" hypothesis that high-degree nodes are more
   * likely to acquire new links, independent of any shared neighborhood.
   *
   * @param adj Adjacency matrix [n, n] (FLOATING_POINT type)
   * @return score Preferential-attachment score matrix [n, n] (FLOATING_POINT type)
   */
  public INDArray preferentialAttachment(INDArray adj) {
    NDValidation.validateFloatingPoint("preferentialAttachment", "adj", adj);
    INDArray deg = Nd4j.base().sum(adj, false, 1);
    INDArray out = Nd4j.base().mmul(Nd4j.base().reshape(deg, -1, 1), Nd4j.base().reshape(deg, 1, -1));
    return out;
  }

  /**
   * Resource-Allocation link-prediction score: S[i,j] = sum_v A[i,v]·A[v,j] / deg_v. Like
   * Adamic-Adar but penalizes high-degree shared neighbors even more strongly (inverse degree
   * rather than inverse log-degree). Often the strongest of the simple topological predictors.
   *
   * @param adj Adjacency matrix [n, n] with positive node degrees (FLOATING_POINT type)
   * @return score Resource-allocation score matrix [n, n] (FLOATING_POINT type)
   */
  public INDArray resourceAllocation(INDArray adj) {
    NDValidation.validateFloatingPoint("resourceAllocation", "adj", adj);
    INDArray deg = Nd4j.base().sum(adj, false, 1);
    INDArray scaled = adj.div(Nd4j.base().reshape(deg, -1, 1));
    INDArray out = Nd4j.base().mmul(adj, scaled);
    return out;
  }

  /**
   * RotatE (Sun et al. 2019): models each relation as an element-wise rotation in complex space
   * (capturing symmetry, inversion and composition).
   * r = (cos(phase), sin(phase)); score = -||head o r - tail||
   *
   * @param hRe Real part of head embeddings [batch, dim] (FLOATING_POINT type)
   * @param hIm Imag part of head embeddings [batch, dim] (FLOATING_POINT type)
   * @param relPhase Relation rotation phases [batch, dim] (radians) (FLOATING_POINT type)
   * @param tRe Real part of tail embeddings [batch, dim] (FLOATING_POINT type)
   * @param tIm Imag part of tail embeddings [batch, dim] (FLOATING_POINT type)
   * @return score Plausibility score [batch] (higher = better) (FLOATING_POINT type)
   */
  public INDArray rotatE(INDArray hRe, INDArray hIm, INDArray relPhase, INDArray tRe,
      INDArray tIm) {
    NDValidation.validateFloatingPoint("rotatE", "hRe", hRe);
    NDValidation.validateFloatingPoint("rotatE", "hIm", hIm);
    NDValidation.validateFloatingPoint("rotatE", "relPhase", relPhase);
    NDValidation.validateFloatingPoint("rotatE", "tRe", tRe);
    NDValidation.validateFloatingPoint("rotatE", "tIm", tIm);
    INDArray cos = Nd4j.math().cos(relPhase);
    INDArray sin = Nd4j.math().sin(relPhase);
    INDArray dRe = hRe.mul(cos).sub(hIm.mul(sin)).sub(tRe);
    INDArray dIm = hRe.mul(sin).add(hIm.mul(cos)).sub(tIm);
    INDArray dist = Nd4j.math().sqrt(Nd4j.base().sum(dRe.mul(dRe).add(dIm.mul(dIm)), false, 1).add(1e-9));
    INDArray out = Nd4j.math().neg(dist);
    return out;
  }

  /**
   * Segment-max pooling: takes max of node embeddings per graph.
   *
   * @param nodeEmb Node embeddings [sumN, F] (FLOATING_POINT type)
   * @param batchVec Node-to-graph index [sumN] (INT type)
   * @return graphEmb Graph-level embeddings [K, F] (FLOATING_POINT type)
   */
  public INDArray segmentMaxPool(INDArray nodeEmb, INDArray batchVec) {
    NDValidation.validateFloatingPoint("segmentMaxPool", "nodeEmb", nodeEmb);
    NDValidation.validateInteger("segmentMaxPool", "batchVec", batchVec);

                INDArray out = Nd4j.base().segmentMax(nodeEmb, batchVec);

    return out;
  }

  /**
   * Segment-mean pooling: produces one embedding per graph from batched node embeddings.
   *
   * @param nodeEmb Node embeddings [sumN, F] (FLOATING_POINT type)
   * @param batchVec Node-to-graph index [sumN] from graphDisjointUnion (INT type)
   * @return graphEmb Graph-level embeddings [K, F] (FLOATING_POINT type)
   */
  public INDArray segmentMeanPool(INDArray nodeEmb, INDArray batchVec) {
    NDValidation.validateFloatingPoint("segmentMeanPool", "nodeEmb", nodeEmb);
    NDValidation.validateInteger("segmentMeanPool", "batchVec", batchVec);

                INDArray out = Nd4j.base().segmentMean(nodeEmb, batchVec);

    return out;
  }

  /**
   * Segment-sum pooling: sums node embeddings per graph.
   *
   * @param nodeEmb Node embeddings [sumN, F] (FLOATING_POINT type)
   * @param batchVec Node-to-graph index [sumN] (INT type)
   * @return graphEmb Graph-level embeddings [K, F] (FLOATING_POINT type)
   */
  public INDArray segmentSumPool(INDArray nodeEmb, INDArray batchVec) {
    NDValidation.validateFloatingPoint("segmentSumPool", "nodeEmb", nodeEmb);
    NDValidation.validateInteger("segmentSumPool", "batchVec", batchVec);

                INDArray out = Nd4j.base().segmentSum(nodeEmb, batchVec);

    return out;
  }

  /**
   * Set2Set graph readout (Vinyals et al. 2016): runs processingSteps rounds of scaled dot-product
   * attention over node embeddings (using the fused sd.nn().dotProductAttentionV2) followed by a GRU
   * state update; produces a permutation-invariant graph-level embedding of size 2d.
   *
   * At each step t the query h [1,d] attends over keys/values nodeEmb [n,d] via
   * dotProductAttentionV2([1,1,d], [1,n,d], [1,n,d]) → attended memory m [1,d], then a GRU cell
   * (same concat-weight form as the library's GGNN for correct CUDA backward) updates h from m.
   * Output = concat(m_T, h_T) [1, 2d].
   *
   * Weights: wZr, wZu [2d, d] (reset / update gate); wC [2d, d] (candidate); bZr, bZu, bC [1, d].
   * Pass qInit = zeros [1,d] at inference; declare as sd.var for end-to-end training.
   *
   * @param nodeEmb Node feature matrix [n, d] (FLOATING_POINT type)
   * @param qInit Initial GRU query state [1, d] (caller passes zeros) (FLOATING_POINT type)
   * @param wZr Reset-gate weights [2d, d] (inSize=d, numUnits=d) (FLOATING_POINT type)
   * @param bZr Reset-gate bias [1, d] (FLOATING_POINT type)
   * @param wZu Update-gate weights [2d, d] (FLOATING_POINT type)
   * @param bZu Update-gate bias [1, d] (FLOATING_POINT type)
   * @param wC Candidate-state weights [2d, d] (FLOATING_POINT type)
   * @param bC Candidate-state bias [1, d] (FLOATING_POINT type)
   * @param processingSteps Number of Set2Set processing steps (T >= 1)
   * @param d Node embedding / GRU hidden dimension
   * @return readout Permutation-invariant graph readout [1, 2d] (FLOATING_POINT type)
   */
  public INDArray set2Set(INDArray nodeEmb, INDArray qInit, INDArray wZr, INDArray bZr,
      INDArray wZu, INDArray bZu, INDArray wC, INDArray bC, int processingSteps, long d) {
    NDValidation.validateFloatingPoint("set2Set", "nodeEmb", nodeEmb);
    NDValidation.validateFloatingPoint("set2Set", "qInit", qInit);
    NDValidation.validateFloatingPoint("set2Set", "wZr", wZr);
    NDValidation.validateFloatingPoint("set2Set", "bZr", bZr);
    NDValidation.validateFloatingPoint("set2Set", "wZu", wZu);
    NDValidation.validateFloatingPoint("set2Set", "bZu", bZu);
    NDValidation.validateFloatingPoint("set2Set", "wC", wC);
    NDValidation.validateFloatingPoint("set2Set", "bC", bC);
    INDArray h = qInit;
    INDArray xKV = Nd4j.base().reshape(nodeEmb, 1L, -1L, d);
    INDArray m = Nd4j.base().zerosLike(qInit);
    for (int t = 0; t < processingSteps; t++) {
        INDArray qQ = Nd4j.base().reshape(h, 1L, 1L, d);
        INDArray attn3 = Nd4j.nn().dotProductAttentionV2(qQ, xKV, xKV, null, null, 0.0, 0.0, false, false);
        m = Nd4j.base().reshape(attn3, 1L, d);
        INDArray xh = Nd4j.base().concat(1, m, h);
        INDArray zr = Nd4j.nn().sigmoid(Nd4j.base().mmul(xh, wZr).add(bZr));
        INDArray zu = Nd4j.nn().sigmoid(Nd4j.base().mmul(xh, wZu).add(bZu));
        INDArray rh = Nd4j.base().concat(1, m, zr.mul(h));
        INDArray hh = Nd4j.math().tanh(Nd4j.base().mmul(rh, wC).add(bC));
        h = h.mul(zu.mul(-1.0).add(1.0)).add(zu.mul(hh));
    }
    INDArray out = Nd4j.base().concat(1, m, h);
    return out;
  }

  /**
   * SimRank node-similarity (Jeh and Widom 2002): two nodes are similar if their in-neighbors are
   * similar. Converges via the fixed-point iteration S_{t+1} = C * W^T S_t W (diagonal forced to 1),
   * where W is the column-normalized adjacency. The diagonal reset is implemented in a gradient-clean
   * elementwise form: S_new = prop * (ones - I) + I.
   * Pass identity = sd.constant(Nd4j.eye(n).castTo(DataType.DOUBLE)) so its gradient is not tracked.
   *
   * @param W Column-normalized adjacency [n, n] (each column sums to 1) (FLOATING_POINT type)
   * @param identity Identity matrix [n, n] -- pass sd.constant(Nd4j.eye(n)) (FLOATING_POINT type)
   * @param C SimRank decay constant in (0, 1)
   * @param iterations Number of power iterations
   * @return out Node-similarity matrix [n, n]; diagonal = 1 (FLOATING_POINT type)
   */
  public INDArray simRank(INDArray W, INDArray identity, double C, int iterations) {
    NDValidation.validateFloatingPoint("simRank", "W", W);
    NDValidation.validateFloatingPoint("simRank", "identity", identity);
    INDArray onesM = Nd4j.base().onesLike(W);
    INDArray offDiagMask = onesM.sub(identity);
    INDArray S = identity;
    for (int i = 0; i < iterations; i++) {
        INDArray prop = Nd4j.base().mmul(Nd4j.base().mmul(W, S, true, false, false), W).mul(C);
        S = prop.mul(offDiagMask).add(identity);
    }
    INDArray out = S;
    return out;
  }

  /**
   * SortPool (Zhang et al. 2018, DGCNN): sorts all n nodes by a scalar sort key (e.g. the last
   * GNN channel) in descending order and keeps the top-k rows, producing a fixed-size graph-level
   * representation [k, d] regardless of graph size. Unlike topKPool there is no sigmoid gate --
   * the selection is purely order-based, which preserves the structural ordering signal used by
   * DGCNN's subsequent 1-D convolution. Differentiable w.r.t. features through the top-k gather
   * (oneHot @ features); the sort key selects WHICH rows are kept but its gradient is zero (as for
   * topKPool / knnGraph -- it enters only through the index, not the value).
   *
   * @param features Node feature matrix [n, d] (FLOATING_POINT type)
   * @param sortKey Per-node sort scores [n] (e.g. last GNN channel) (FLOATING_POINT type)
   * @param k Number of nodes to keep (output rows)
   * @param n Number of nodes (one-hot depth)
   * @return pooled Top-k node features in descending sort-key order [k, d] (FLOATING_POINT type)
   */
  public INDArray sortPool(INDArray features, INDArray sortKey, int k, int n) {
    NDValidation.validateFloatingPoint("sortPool", "features", features);
    NDValidation.validateFloatingPoint("sortPool", "sortKey", sortKey);
    INDArray[] tk = Nd4j.nn().topK(sortKey, k, true);
    INDArray sel = Nd4j.base().oneHot(tk[1], n, -1, 1.0, 0.0, features.dataType());
    INDArray out = Nd4j.base().mmul(sel, features);
    return out;
  }

  /**
   * Top-k node pooling (Gao and Ji 2019 / Cangea et al. 2018; the selection mechanism of SAGPool):
   * keeps the k highest-scoring nodes and gates their features by sigmoid(score) so the score stays
   * differentiable. The top-k rows are gathered as oneHot(topIndices, n) @ features (avoiding a
   * separate gather op). Pass scores from a learned projection or a graph-attention layer for SAGPool.
   *
   * @param scores Per-node selection scores [n] (e.g. a learned projection) (FLOATING_POINT type)
   * @param features Node features [n, d] (FLOATING_POINT type)
   * @param k Number of nodes to keep
   * @param n Number of nodes (the one-hot depth)
   * @return pooled Pooled features of the top-k nodes [k, d], gated by sigmoid(score) (FLOATING_POINT type)
   */
  public INDArray topKPool(INDArray scores, INDArray features, int k, int n) {
    NDValidation.validateFloatingPoint("topKPool", "scores", scores);
    NDValidation.validateFloatingPoint("topKPool", "features", features);
    INDArray[] tk = Nd4j.nn().topK(scores, k, false);
    INDArray sel = Nd4j.base().oneHot(tk[1], n, -1, 1.0, 0.0, features.dataType());
    INDArray gathered = Nd4j.base().mmul(sel, features);
    INDArray gate = Nd4j.nn().sigmoid(tk[0]);
    INDArray out = gathered.mul(Nd4j.base().expandDims(gate, -1));
    return out;
  }

  /**
   * TransE (Bordes et al. 2013): models a relation as a translation, h + r ~ t.
   * score = -||head + relation - tail||
   *
   * @param head Head-entity embeddings [batch, dim] (FLOATING_POINT type)
   * @param relation Relation embeddings [batch, dim] (FLOATING_POINT type)
   * @param tail Tail-entity embeddings [batch, dim] (FLOATING_POINT type)
   * @return score Plausibility score [batch] (higher = better) (FLOATING_POINT type)
   */
  public INDArray transE(INDArray head, INDArray relation, INDArray tail) {
    NDValidation.validateFloatingPoint("transE", "head", head);
    NDValidation.validateFloatingPoint("transE", "relation", relation);
    NDValidation.validateFloatingPoint("transE", "tail", tail);
    INDArray diff = head.add(relation).sub(tail);
    INDArray dist = Nd4j.math().sqrt(Nd4j.base().sum(diff.mul(diff), false, 1).add(1e-9));
    INDArray out = Nd4j.math().neg(dist);
    return out;
  }

  /**
   * Time-aware TransE (TTransE, Jiang et al. 2016) for temporal knowledge graphs: the timestamp
   * embedding is an additional translation.
   * score = -||head + relation + time - tail||
   *
   * @param head Head-entity embeddings [batch, dim] (FLOATING_POINT type)
   * @param relation Relation embeddings [batch, dim] (FLOATING_POINT type)
   * @param time Timestamp embeddings [batch, dim] (FLOATING_POINT type)
   * @param tail Tail-entity embeddings [batch, dim] (FLOATING_POINT type)
   * @return score Plausibility score [batch] (higher = better) (FLOATING_POINT type)
   */
  public INDArray transET(INDArray head, INDArray relation, INDArray time, INDArray tail) {
    NDValidation.validateFloatingPoint("transET", "head", head);
    NDValidation.validateFloatingPoint("transET", "relation", relation);
    NDValidation.validateFloatingPoint("transET", "time", time);
    NDValidation.validateFloatingPoint("transET", "tail", tail);
    INDArray diff = head.add(relation).add(time).sub(tail);
    INDArray dist = Nd4j.math().sqrt(Nd4j.base().sum(diff.mul(diff), false, 1).add(1e-9));
    INDArray out = Nd4j.math().neg(dist);
    return out;
  }

  /**
   * TransH (Wang et al. 2014): like TransE, but head and tail are projected onto a relation-specific
   * hyperplane (normal wr), so an entity can play different roles under different relations.
   * score = -||hPerp + relation - tPerp||
   *
   * @param head Head-entity embeddings [batch, dim] (FLOATING_POINT type)
   * @param wr Relation hyperplane normals [batch, dim] (ideally unit-norm) (FLOATING_POINT type)
   * @param relation Relation translation embeddings [batch, dim] (FLOATING_POINT type)
   * @param tail Tail-entity embeddings [batch, dim] (FLOATING_POINT type)
   * @return score Plausibility score [batch] (higher = better) (FLOATING_POINT type)
   */
  public INDArray transH(INDArray head, INDArray wr, INDArray relation, INDArray tail) {
    NDValidation.validateFloatingPoint("transH", "head", head);
    NDValidation.validateFloatingPoint("transH", "wr", wr);
    NDValidation.validateFloatingPoint("transH", "relation", relation);
    NDValidation.validateFloatingPoint("transH", "tail", tail);
    INDArray hPerp = head.sub(wr.mul(Nd4j.base().sum(wr.mul(head), true, 1)));
    INDArray tPerp = tail.sub(wr.mul(Nd4j.base().sum(wr.mul(tail), true, 1)));
    INDArray diff = hPerp.add(relation).sub(tPerp);
    INDArray dist = Nd4j.math().sqrt(Nd4j.base().sum(diff.mul(diff), false, 1).add(1e-9));
    INDArray out = Nd4j.math().neg(dist);
    return out;
  }

  /**
   * TuckER (Balazevic et al. 2019): a Tucker-decomposition bilinear model with a learnable core
   * tensor shared across all triples; subsumes DistMult / ComplEx / SimplE.
   * score = W x1 head x2 relation x3 tail
   *
   * @param head Head-entity embeddings [batch, de] (FLOATING_POINT type)
   * @param relation Relation embeddings [batch, dr] (FLOATING_POINT type)
   * @param tail Tail-entity embeddings [batch, de] (FLOATING_POINT type)
   * @param coreW Core tensor [de, dr, de] (FLOATING_POINT type)
   * @param de Entity embedding dimension
   * @param dr Relation embedding dimension
   * @return score Plausibility score [batch] (higher = better) (FLOATING_POINT type)
   */
  public INDArray tuckER(INDArray head, INDArray relation, INDArray tail, INDArray coreW, int de,
      int dr) {
    NDValidation.validateFloatingPoint("tuckER", "head", head);
    NDValidation.validateFloatingPoint("tuckER", "relation", relation);
    NDValidation.validateFloatingPoint("tuckER", "tail", tail);
    NDValidation.validateFloatingPoint("tuckER", "coreW", coreW);
    INDArray coreUnfold = Nd4j.base().reshape(coreW, de, dr * de);
    INDArray m1 = Nd4j.base().reshape(Nd4j.base().mmul(head, coreUnfold), -1, dr, de);
    INDArray rExp = Nd4j.base().reshape(relation, -1, dr, 1);
    INDArray m2 = Nd4j.base().sum(rExp.mul(m1), false, 1);
    INDArray out = Nd4j.base().sum(m2.mul(tail), false, 1);
    return out;
  }
}
