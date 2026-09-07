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

package org.nd4j.autodiff.samediff.ops;

import static org.nd4j.autodiff.samediff.ops.SDValidation.isSameType;

import java.lang.String;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;

public class SDGraph extends SDOps {
  public SDGraph(SameDiff sameDiff) {
    super(sameDiff);
  }

  /**
   * Adamic-Adar link-prediction score: S[i,j] = sum_v A[i,v]·A[v,j] / log(deg_v), weighting each
   * shared neighbor v by the inverse log of its degree so that rare (low-degree) common neighbors
   * contribute more than hubs. Assumes node degrees > 1 (so log(deg) > 0).
   *
   * @param adj Adjacency matrix [n, n] with node degrees > 1 (FLOATING_POINT type)
   * @return score Adamic-Adar score matrix [n, n] (FLOATING_POINT type)
   */
  public SDVariable adamicAdar(SDVariable adj) {
    SDValidation.validateFloatingPoint("adamicAdar", "adj", adj);
    SDVariable deg = sd.sum(adj, false, 1);
    SDVariable logDeg = sd.math().log(deg);
    SDVariable scaled = adj.div(sd.reshape(logDeg, -1, 1));
    SDVariable out = sd.mmul(adj, scaled);
    return out;
  }

  /**
   * Adamic-Adar link-prediction score: S[i,j] = sum_v A[i,v]·A[v,j] / log(deg_v), weighting each
   * shared neighbor v by the inverse log of its degree so that rare (low-degree) common neighbors
   * contribute more than hubs. Assumes node degrees > 1 (so log(deg) > 0).
   *
   * @param name name May be null. Name for the output variable
   * @param adj Adjacency matrix [n, n] with node degrees > 1 (FLOATING_POINT type)
   * @return score Adamic-Adar score matrix [n, n] (FLOATING_POINT type)
   */
  public SDVariable adamicAdar(String name, SDVariable adj) {
    SDValidation.validateFloatingPoint("adamicAdar", "adj", adj);
    SDVariable deg = sd.sum(adj, false, 1);
    SDVariable logDeg = sd.math().log(deg);
    SDVariable scaled = adj.div(sd.reshape(logDeg, -1, 1));
    SDVariable out = sd.mmul(adj, scaled);
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable bgrlLoss(SDVariable onlineZ, SDVariable targetZ, SDVariable predW) {
    SDValidation.validateFloatingPoint("bgrlLoss", "onlineZ", onlineZ);
    SDValidation.validateFloatingPoint("bgrlLoss", "targetZ", targetZ);
    SDValidation.validateFloatingPoint("bgrlLoss", "predW", predW);
    SDVariable pred = sd.mmul(onlineZ, predW);
    SDVariable predNorm = pred.div(sd.math().sqrt(sd.sum(pred.mul(pred), true, 1).add(1e-12)));
    SDVariable tNorm = targetZ.div(sd.math().sqrt(sd.sum(targetZ.mul(targetZ), true, 1).add(1e-12)));
    SDVariable cos = sd.sum(predNorm.mul(tNorm), false, 1);
    SDVariable out = sd.mean(cos.mul(-2.0).add(2.0));
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
   * @param name name May be null. Name for the output variable
   * @param onlineZ Online encoder node embeddings [n, d] (FLOATING_POINT type)
   * @param targetZ Target (stop-grad) node embeddings [n, d] (FLOATING_POINT type)
   * @param predW Online predictor weight matrix [d, d] (FLOATING_POINT type)
   * @return loss Scalar BGRL loss: mean(2 - 2·cosine(onlineZ·predW, targetZ)) (FLOATING_POINT type)
   */
  public SDVariable bgrlLoss(String name, SDVariable onlineZ, SDVariable targetZ,
      SDVariable predW) {
    SDValidation.validateFloatingPoint("bgrlLoss", "onlineZ", onlineZ);
    SDValidation.validateFloatingPoint("bgrlLoss", "targetZ", targetZ);
    SDValidation.validateFloatingPoint("bgrlLoss", "predW", predW);
    SDVariable pred = sd.mmul(onlineZ, predW);
    SDVariable predNorm = pred.div(sd.math().sqrt(sd.sum(pred.mul(pred), true, 1).add(1e-12)));
    SDVariable tNorm = targetZ.div(sd.math().sqrt(sd.sum(targetZ.mul(targetZ), true, 1).add(1e-12)));
    SDVariable cos = sd.sum(predNorm.mul(tNorm), false, 1);
    SDVariable out = sd.mean(cos.mul(-2.0).add(2.0));
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable clusteringCoefficient(SDVariable adj, SDVariable identity) {
    SDValidation.validateFloatingPoint("clusteringCoefficient", "adj", adj);
    SDValidation.validateFloatingPoint("clusteringCoefficient", "identity", identity);
    SDVariable a2 = sd.mmul(adj, adj);
    SDVariable a3 = sd.mmul(a2, adj);
    SDVariable tri = sd.sum(a3.mul(identity), false, 1);
    SDVariable deg = sd.sum(adj, false, 1);
    SDVariable denom = deg.mul(deg.sub(1.0)).add(1e-9);
    SDVariable out = tri.div(denom);
    return out;
  }

  /**
   * Local clustering coefficient per node: C_i = (closed triangles through i) / (possible pairs) =
   * (A^3)_ii / (deg_i (deg_i - 1)). (A^3)_ii counts the length-3 closed walks through node i (= twice
   * its triangle count), computed as diag(A·A·A) via an identity-mask extraction. Measures how
   * tightly each node's neighborhood is interconnected. Pass identity = sd.constant(Nd4j.eye(n)).
   *
   * @param name name May be null. Name for the output variable
   * @param adj Adjacency matrix [n, n] (symmetric for undirected graphs) (FLOATING_POINT type)
   * @param identity Identity matrix [n, n] -- pass sd.constant(Nd4j.eye(n)) (FLOATING_POINT type)
   * @return coeff Per-node local clustering coefficient [n] (FLOATING_POINT type)
   */
  public SDVariable clusteringCoefficient(String name, SDVariable adj, SDVariable identity) {
    SDValidation.validateFloatingPoint("clusteringCoefficient", "adj", adj);
    SDValidation.validateFloatingPoint("clusteringCoefficient", "identity", identity);
    SDVariable a2 = sd.mmul(adj, adj);
    SDVariable a3 = sd.mmul(a2, adj);
    SDVariable tri = sd.sum(a3.mul(identity), false, 1);
    SDVariable deg = sd.sum(adj, false, 1);
    SDVariable denom = deg.mul(deg.sub(1.0)).add(1e-9);
    SDVariable out = tri.div(denom);
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Common-Neighbors link-prediction score: S = A·A, so S[i,j] counts the (weighted) paths of
   * length two between i and j -- the number of neighbors they share. The simplest topological
   * link predictor; higher scores indicate more likely missing edges.
   *
   * @param adj Adjacency matrix [n, n] (symmetric for undirected graphs) (FLOATING_POINT type)
   * @return score Common-neighbor score matrix [n, n] (FLOATING_POINT type)
   */
  public SDVariable commonNeighbors(SDVariable adj) {
    SDValidation.validateFloatingPoint("commonNeighbors", "adj", adj);
    SDVariable out = sd.mmul(adj, adj);
    return out;
  }

  /**
   * Common-Neighbors link-prediction score: S = A·A, so S[i,j] counts the (weighted) paths of
   * length two between i and j -- the number of neighbors they share. The simplest topological
   * link predictor; higher scores indicate more likely missing edges.
   *
   * @param name name May be null. Name for the output variable
   * @param adj Adjacency matrix [n, n] (symmetric for undirected graphs) (FLOATING_POINT type)
   * @return score Common-neighbor score matrix [n, n] (FLOATING_POINT type)
   */
  public SDVariable commonNeighbors(String name, SDVariable adj) {
    SDValidation.validateFloatingPoint("commonNeighbors", "adj", adj);
    SDVariable out = sd.mmul(adj, adj);
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable complEx(SDVariable hRe, SDVariable hIm, SDVariable rRe, SDVariable rIm,
      SDVariable tRe, SDVariable tIm) {
    SDValidation.validateFloatingPoint("complEx", "hRe", hRe);
    SDValidation.validateFloatingPoint("complEx", "hIm", hIm);
    SDValidation.validateFloatingPoint("complEx", "rRe", rRe);
    SDValidation.validateFloatingPoint("complEx", "rIm", rIm);
    SDValidation.validateFloatingPoint("complEx", "tRe", tRe);
    SDValidation.validateFloatingPoint("complEx", "tIm", tIm);
    SDVariable s = hRe.mul(rRe).mul(tRe).add(hRe.mul(rIm).mul(tIm)).add(hIm.mul(rRe).mul(tIm)).sub(hIm.mul(rIm).mul(tRe));
    SDVariable out = sd.sum(s, false, 1);
    return out;
  }

  /**
   * ComplEx (Trouillon et al. 2016): a complex-valued bilinear product whose real part scores the
   * triple; the imaginary parts let it model asymmetric relations.
   * score = Re( sum_d head_d * relation_d * conj(tail_d) )
   *
   * @param name name May be null. Name for the output variable
   * @param hRe Real part of head embeddings [batch, dim] (FLOATING_POINT type)
   * @param hIm Imag part of head embeddings [batch, dim] (FLOATING_POINT type)
   * @param rRe Real part of relation embeddings [batch, dim] (FLOATING_POINT type)
   * @param rIm Imag part of relation embeddings [batch, dim] (FLOATING_POINT type)
   * @param tRe Real part of tail embeddings [batch, dim] (FLOATING_POINT type)
   * @param tIm Imag part of tail embeddings [batch, dim] (FLOATING_POINT type)
   * @return score Plausibility score [batch] (higher = better) (FLOATING_POINT type)
   */
  public SDVariable complEx(String name, SDVariable hRe, SDVariable hIm, SDVariable rRe,
      SDVariable rIm, SDVariable tRe, SDVariable tIm) {
    SDValidation.validateFloatingPoint("complEx", "hRe", hRe);
    SDValidation.validateFloatingPoint("complEx", "hIm", hIm);
    SDValidation.validateFloatingPoint("complEx", "rRe", rRe);
    SDValidation.validateFloatingPoint("complEx", "rIm", rIm);
    SDValidation.validateFloatingPoint("complEx", "tRe", tRe);
    SDValidation.validateFloatingPoint("complEx", "tIm", tIm);
    SDVariable s = hRe.mul(rRe).mul(tRe).add(hRe.mul(rIm).mul(tIm)).add(hIm.mul(rRe).mul(tIm)).sub(hIm.mul(rIm).mul(tRe));
    SDVariable out = sd.sum(s, false, 1);
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable convE(SDVariable head, SDVariable relation, SDVariable tail, SDVariable convW,
      SDVariable convB, SDVariable fcW, SDVariable fcB, int embH, int embW, int channels) {
    SDValidation.validateFloatingPoint("convE", "head", head);
    SDValidation.validateFloatingPoint("convE", "relation", relation);
    SDValidation.validateFloatingPoint("convE", "tail", tail);
    SDValidation.validateFloatingPoint("convE", "convW", convW);
    SDValidation.validateFloatingPoint("convE", "convB", convB);
    SDValidation.validateFloatingPoint("convE", "fcW", fcW);
    SDValidation.validateFloatingPoint("convE", "fcB", fcB);
    SDVariable hImg = sd.reshape(head, -1, embH, embW);
    SDVariable rImg = sd.reshape(relation, -1, embH, embW);
    SDVariable stacked = sd.concat(1, hImg, rImg);
    SDVariable img = sd.reshape(stacked, -1, 1, 2L * embH, embW);
    Conv2DConfig cfg = Conv2DConfig.builder().kH(3).kW(3).build();
    SDVariable feat = sd.nn().relu(sd.cnn().conv2d(img, convW, convB, cfg), 0.0);
    long flatDim = (long) channels * (2L * embH - 2) * (embW - 2);
    SDVariable flat = sd.reshape(feat, -1, flatDim);
    SDVariable fc = sd.nn().relu(sd.mmul(flat, fcW).add(fcB), 0.0);
    SDVariable out = sd.sum(fc.mul(tail), false, 1);
    return out;
  }

  /**
   * ConvE (Dettmers et al. 2018): reshape head + relation into 2D images, stack, run a 2D
   * convolution + fully-connected projection, then score against the tail. A strong,
   * parameter-efficient KGE baseline.
   *
   * @param name name May be null. Name for the output variable
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
  public SDVariable convE(String name, SDVariable head, SDVariable relation, SDVariable tail,
      SDVariable convW, SDVariable convB, SDVariable fcW, SDVariable fcB, int embH, int embW,
      int channels) {
    SDValidation.validateFloatingPoint("convE", "head", head);
    SDValidation.validateFloatingPoint("convE", "relation", relation);
    SDValidation.validateFloatingPoint("convE", "tail", tail);
    SDValidation.validateFloatingPoint("convE", "convW", convW);
    SDValidation.validateFloatingPoint("convE", "convB", convB);
    SDValidation.validateFloatingPoint("convE", "fcW", fcW);
    SDValidation.validateFloatingPoint("convE", "fcB", fcB);
    SDVariable hImg = sd.reshape(head, -1, embH, embW);
    SDVariable rImg = sd.reshape(relation, -1, embH, embW);
    SDVariable stacked = sd.concat(1, hImg, rImg);
    SDVariable img = sd.reshape(stacked, -1, 1, 2L * embH, embW);
    Conv2DConfig cfg = Conv2DConfig.builder().kH(3).kW(3).build();
    SDVariable feat = sd.nn().relu(sd.cnn().conv2d(img, convW, convB, cfg), 0.0);
    long flatDim = (long) channels * (2L * embH - 2) * (embW - 2);
    SDVariable flat = sd.reshape(feat, -1, flatDim);
    SDVariable fc = sd.nn().relu(sd.mmul(flat, fcW).add(fcB), 0.0);
    SDVariable out = sd.sum(fc.mul(tail), false, 1);
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable correctAndSmooth(SDVariable basePreds, SDVariable aNorm, SDVariable residuals,
      double alpha1, double alpha2, int iter1, int iter2) {
    SDValidation.validateFloatingPoint("correctAndSmooth", "basePreds", basePreds);
    SDValidation.validateFloatingPoint("correctAndSmooth", "aNorm", aNorm);
    SDValidation.validateFloatingPoint("correctAndSmooth", "residuals", residuals);
    SDVariable E = residuals;
    for (int i = 0; i < iter1; i++) {
        E = sd.mmul(aNorm, E).mul(alpha1).add(residuals.mul(1.0 - alpha1));
    }
    SDVariable corrected = basePreds.add(E);
    SDVariable Y = corrected;
    for (int j = 0; j < iter2; j++) {
        Y = sd.mmul(aNorm, Y).mul(alpha2).add(corrected.mul(1.0 - alpha2));
    }
    SDVariable out = Y;
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
   * @param name name May be null. Name for the output variable
   * @param basePreds Base per-node predictions / logits [n, c] (FLOATING_POINT type)
   * @param aNorm Dense (row- or sym-) normalized adjacency [n, n] (FLOATING_POINT type)
   * @param residuals Label residuals at training nodes, zeros elsewhere [n, c] (FLOATING_POINT type)
   * @param alpha1 Residual-spreading weight in (0,1) for the Correct phase
   * @param alpha2 Label-smoothing weight in (0,1) for the Smooth phase
   * @param iter1 Number of Correct-phase propagation steps
   * @param iter2 Number of Smooth-phase propagation steps
   * @return out Corrected and smoothed per-node predictions [n, c] (FLOATING_POINT type)
   */
  public SDVariable correctAndSmooth(String name, SDVariable basePreds, SDVariable aNorm,
      SDVariable residuals, double alpha1, double alpha2, int iter1, int iter2) {
    SDValidation.validateFloatingPoint("correctAndSmooth", "basePreds", basePreds);
    SDValidation.validateFloatingPoint("correctAndSmooth", "aNorm", aNorm);
    SDValidation.validateFloatingPoint("correctAndSmooth", "residuals", residuals);
    SDVariable E = residuals;
    for (int i = 0; i < iter1; i++) {
        E = sd.mmul(aNorm, E).mul(alpha1).add(residuals.mul(1.0 - alpha1));
    }
    SDVariable corrected = basePreds.add(E);
    SDVariable Y = corrected;
    for (int j = 0; j < iter2; j++) {
        Y = sd.mmul(aNorm, Y).mul(alpha2).add(corrected.mul(1.0 - alpha2));
    }
    SDVariable out = Y;
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Pearson correlation matrix among the columns (variables) of a data matrix -- the basis of a
   * correlation graph. Threshold |corr| with sd.sparse().denseToCsr(...) to obtain a sparse
   * correlation-graph adjacency.
   *
   * @param data Observations x variables [n, d] (FLOATING_POINT type)
   * @return corr Pearson correlation matrix [d, d] with unit diagonal (FLOATING_POINT type)
   */
  public SDVariable correlationMatrix(SDVariable data) {
    SDValidation.validateFloatingPoint("correlationMatrix", "data", data);
    SDVariable mean = sd.mean(data, true, 0);
    SDVariable centered = data.sub(mean);
    SDVariable cov = sd.mmul(centered, centered, true, false, false);
    SDVariable variance = sd.sum(centered.mul(centered), false, 0);
    SDVariable std = sd.math().sqrt(variance.add(1e-12));
    SDVariable denom = sd.mmul(sd.reshape(std, -1, 1), sd.reshape(std, 1, -1));
    SDVariable out = cov.div(denom);
    return out;
  }

  /**
   * Pearson correlation matrix among the columns (variables) of a data matrix -- the basis of a
   * correlation graph. Threshold |corr| with sd.sparse().denseToCsr(...) to obtain a sparse
   * correlation-graph adjacency.
   *
   * @param name name May be null. Name for the output variable
   * @param data Observations x variables [n, d] (FLOATING_POINT type)
   * @return corr Pearson correlation matrix [d, d] with unit diagonal (FLOATING_POINT type)
   */
  public SDVariable correlationMatrix(String name, SDVariable data) {
    SDValidation.validateFloatingPoint("correlationMatrix", "data", data);
    SDVariable mean = sd.mean(data, true, 0);
    SDVariable centered = data.sub(mean);
    SDVariable cov = sd.mmul(centered, centered, true, false, false);
    SDVariable variance = sd.sum(centered.mul(centered), false, 0);
    SDVariable std = sd.math().sqrt(variance.add(1e-12));
    SDVariable denom = sd.mmul(sd.reshape(std, -1, 1), sd.reshape(std, 1, -1));
    SDVariable out = cov.div(denom);
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Cosine-similarity affinity between the rows (nodes) of a feature matrix.
   * sim[i,j] = (x_i . x_j) / (||x_i|| ||x_j||). A standard input to a kNN / thresholded similarity graph.
   *
   * @param features Node features [n, d] (FLOATING_POINT type)
   * @return sim Cosine-similarity matrix [n, n] with unit diagonal (FLOATING_POINT type)
   */
  public SDVariable cosineSimilarity(SDVariable features) {
    SDValidation.validateFloatingPoint("cosineSimilarity", "features", features);
    SDVariable norm = sd.math().sqrt(sd.sum(features.mul(features), true, 1).add(1e-12));
    SDVariable normalized = features.div(norm);
    SDVariable out = sd.mmul(normalized, normalized, false, true, false);
    return out;
  }

  /**
   * Cosine-similarity affinity between the rows (nodes) of a feature matrix.
   * sim[i,j] = (x_i . x_j) / (||x_i|| ||x_j||). A standard input to a kNN / thresholded similarity graph.
   *
   * @param name name May be null. Name for the output variable
   * @param features Node features [n, d] (FLOATING_POINT type)
   * @return sim Cosine-similarity matrix [n, n] with unit diagonal (FLOATING_POINT type)
   */
  public SDVariable cosineSimilarity(String name, SDVariable features) {
    SDValidation.validateFloatingPoint("cosineSimilarity", "features", features);
    SDVariable norm = sd.math().sqrt(sd.sum(features.mul(features), true, 1).add(1e-12));
    SDVariable normalized = features.div(norm);
    SDVariable out = sd.mmul(normalized, normalized, false, true, false);
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable dgiLoss(SDVariable H, SDVariable Hneg, SDVariable discW) {
    SDValidation.validateFloatingPoint("dgiLoss", "H", H);
    SDValidation.validateFloatingPoint("dgiLoss", "Hneg", Hneg);
    SDValidation.validateFloatingPoint("dgiLoss", "discW", discW);
    SDVariable s = sd.nn().sigmoid(sd.mean(H, true, 0));
    SDVariable sW = sd.mmul(s, discW, false, true, false);
    SDVariable pos = sd.sum(H.mul(sW), false, 1);
    SDVariable neg = sd.sum(Hneg.mul(sW), false, 1);
    SDVariable posTerm = sd.math().log(sd.nn().sigmoid(pos).add(1e-12));
    SDVariable negTerm = sd.math().log(sd.nn().sigmoid(neg.mul(-1.0)).add(1e-12));
    SDVariable out = sd.mean(posTerm.add(negTerm), false).mul(-1.0);
    return out;
  }

  /**
   * Deep Graph Infomax loss (Velickovic et al. 2019): maximizes mutual information between each
   * node's encoding and a global graph summary via a bilinear discriminator that tells real node
   * encodings from encodings of a corrupted graph. Yields label-free node embeddings.
   *
   * @param name name May be null. Name for the output variable
   * @param H Node encodings of the real graph [n, d] (FLOATING_POINT type)
   * @param Hneg Node encodings of the corrupted graph [n, d] (FLOATING_POINT type)
   * @param discW Bilinear discriminator weight [d, d] (FLOATING_POINT type)
   * @return loss Scalar DGI loss (FLOATING_POINT type)
   */
  public SDVariable dgiLoss(String name, SDVariable H, SDVariable Hneg, SDVariable discW) {
    SDValidation.validateFloatingPoint("dgiLoss", "H", H);
    SDValidation.validateFloatingPoint("dgiLoss", "Hneg", Hneg);
    SDValidation.validateFloatingPoint("dgiLoss", "discW", discW);
    SDVariable s = sd.nn().sigmoid(sd.mean(H, true, 0));
    SDVariable sW = sd.mmul(s, discW, false, true, false);
    SDVariable pos = sd.sum(H.mul(sW), false, 1);
    SDVariable neg = sd.sum(Hneg.mul(sW), false, 1);
    SDVariable posTerm = sd.math().log(sd.nn().sigmoid(pos).add(1e-12));
    SDVariable negTerm = sd.math().log(sd.nn().sigmoid(neg.mul(-1.0)).add(1e-12));
    SDVariable out = sd.mean(posTerm.add(negTerm), false).mul(-1.0);
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable distMult(SDVariable head, SDVariable relation, SDVariable tail) {
    SDValidation.validateFloatingPoint("distMult", "head", head);
    SDValidation.validateFloatingPoint("distMult", "relation", relation);
    SDValidation.validateFloatingPoint("distMult", "tail", tail);
    SDVariable out = sd.sum(head.mul(relation).mul(tail), false, 1);
    return out;
  }

  /**
   * DistMult (Yang et al. 2015): a symmetric trilinear product.
   * score = sum_d head_d * relation_d * tail_d
   *
   * @param name name May be null. Name for the output variable
   * @param head Head-entity embeddings [batch, dim] (FLOATING_POINT type)
   * @param relation Relation embeddings [batch, dim] (FLOATING_POINT type)
   * @param tail Tail-entity embeddings [batch, dim] (FLOATING_POINT type)
   * @return score Plausibility score [batch] (higher = better) (FLOATING_POINT type)
   */
  public SDVariable distMult(String name, SDVariable head, SDVariable relation, SDVariable tail) {
    SDValidation.validateFloatingPoint("distMult", "head", head);
    SDValidation.validateFloatingPoint("distMult", "relation", relation);
    SDValidation.validateFloatingPoint("distMult", "tail", tail);
    SDVariable out = sd.sum(head.mul(relation).mul(tail), false, 1);
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Gaussian (RBF) similarity affinity between the rows (nodes) of a feature matrix.
   * sim[i,j] = exp( -||x_i - x_j||^2 / (2 sigma^2) ). Threshold or take per-row top-k for a kNN graph.
   *
   * @param features Node features [n, d] (FLOATING_POINT type)
   * @param sigma RBF kernel bandwidth (> 0)
   * @return sim Gaussian-similarity matrix [n, n] with unit diagonal (FLOATING_POINT type)
   */
  public SDVariable gaussianSimilarity(SDVariable features, double sigma) {
    SDValidation.validateFloatingPoint("gaussianSimilarity", "features", features);
    SDVariable xi = sd.expandDims(features, 1);
    SDVariable xj = sd.expandDims(features, 0);
    SDVariable diff = xi.sub(xj);
    SDVariable d2 = sd.sum(diff.mul(diff), false, 2);
    SDVariable out = sd.math().exp(d2.mul(-1.0 / (2.0 * sigma * sigma)));
    return out;
  }

  /**
   * Gaussian (RBF) similarity affinity between the rows (nodes) of a feature matrix.
   * sim[i,j] = exp( -||x_i - x_j||^2 / (2 sigma^2) ). Threshold or take per-row top-k for a kNN graph.
   *
   * @param name name May be null. Name for the output variable
   * @param features Node features [n, d] (FLOATING_POINT type)
   * @param sigma RBF kernel bandwidth (> 0)
   * @return sim Gaussian-similarity matrix [n, n] with unit diagonal (FLOATING_POINT type)
   */
  public SDVariable gaussianSimilarity(String name, SDVariable features, double sigma) {
    SDValidation.validateFloatingPoint("gaussianSimilarity", "features", features);
    SDVariable xi = sd.expandDims(features, 1);
    SDVariable xj = sd.expandDims(features, 0);
    SDVariable diff = xi.sub(xj);
    SDVariable d2 = sd.sum(diff.mul(diff), false, 2);
    SDVariable out = sd.math().exp(d2.mul(-1.0 / (2.0 * sigma * sigma)));
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable graceLoss(SDVariable z1, SDVariable z2, SDVariable identity, double tau) {
    SDValidation.validateFloatingPoint("graceLoss", "z1", z1);
    SDValidation.validateFloatingPoint("graceLoss", "z2", z2);
    SDValidation.validateFloatingPoint("graceLoss", "identity", identity);
    SDVariable z1n = z1.div(sd.math().sqrt(sd.sum(z1.mul(z1), true, 1).add(1e-12)));
    SDVariable z2n = z2.div(sd.math().sqrt(sd.sum(z2.mul(z2), true, 1).add(1e-12)));
    SDVariable sim = sd.mmul(z1n, z2n, false, true, false).div(tau);
    SDVariable pos = sd.sum(sim.mul(identity), false, 1);
    SDVariable lse = sd.math().log(sd.sum(sd.math().exp(sim), false, 1).add(1e-12));
    SDVariable out = sd.mean(lse.sub(pos));
    return out;
  }

  /**
   * GRACE graph-contrastive loss (Zhu et al. 2020): an InfoNCE objective over two augmented views.
   * For each node i, the positive pair is (z1_i, z2_i) and the negatives are all z2_j (j != i):
   * loss = -mean( cosine(z1_i,z2_i)/tau - logsumexp_j cosine(z1_i,z2_j)/tau ).
   * Yields label-free node embeddings. Pass identity = sd.constant(Nd4j.eye(n).castTo(DataType.DOUBLE))
   * so the diagonal (positive) similarities are extracted with a gradient-clean elementwise mask.
   *
   * @param name name May be null. Name for the output variable
   * @param z1 Node embeddings of augmented view 1 [n, d] (FLOATING_POINT type)
   * @param z2 Node embeddings of augmented view 2 [n, d] (FLOATING_POINT type)
   * @param identity Identity matrix [n, n] -- pass sd.constant(Nd4j.eye(n)) (FLOATING_POINT type)
   * @param tau Temperature (e.g. 0.5)
   * @return loss Scalar GRACE / InfoNCE contrastive loss (FLOATING_POINT type)
   */
  public SDVariable graceLoss(String name, SDVariable z1, SDVariable z2, SDVariable identity,
      double tau) {
    SDValidation.validateFloatingPoint("graceLoss", "z1", z1);
    SDValidation.validateFloatingPoint("graceLoss", "z2", z2);
    SDValidation.validateFloatingPoint("graceLoss", "identity", identity);
    SDVariable z1n = z1.div(sd.math().sqrt(sd.sum(z1.mul(z1), true, 1).add(1e-12)));
    SDVariable z2n = z2.div(sd.math().sqrt(sd.sum(z2.mul(z2), true, 1).add(1e-12)));
    SDVariable sim = sd.mmul(z1n, z2n, false, true, false).div(tau);
    SDVariable pos = sd.sum(sim.mul(identity), false, 1);
    SDVariable lse = sd.math().log(sd.sum(sd.math().exp(sim), false, 1).add(1e-12));
    SDVariable out = sd.mean(lse.sub(pos));
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable[] graphDisjointUnion(SDVariable[] Xs, SDVariable[] vals, SDVariable[] colIdxs,
      SDVariable... rowPtrs) {
    SDValidation.validateFloatingPoint("graphDisjointUnion", "Xs", Xs);
    Preconditions.checkArgument(Xs.length >= 1, "Xs has incorrect size/length. Expected: Xs.length >= 1, got %s", Xs.length);
    SDValidation.validateFloatingPoint("graphDisjointUnion", "vals", vals);
    Preconditions.checkArgument(vals.length >= 1, "vals has incorrect size/length. Expected: vals.length >= 1, got %s", vals.length);
    SDValidation.validateInteger("graphDisjointUnion", "colIdxs", colIdxs);
    Preconditions.checkArgument(colIdxs.length >= 1, "colIdxs has incorrect size/length. Expected: colIdxs.length >= 1, got %s", colIdxs.length);
    SDValidation.validateInteger("graphDisjointUnion", "rowPtrs", rowPtrs);
    Preconditions.checkArgument(rowPtrs.length >= 1, "rowPtrs has incorrect size/length. Expected: rowPtrs.length >= 1, got %s", rowPtrs.length);
    return new org.nd4j.linalg.api.ops.impl.sparse.GraphDisjointUnion(sd,Xs, vals, colIdxs, rowPtrs).outputVariables();
  }

  /**
   * Assembles K variable-size graphs into one block-diagonal graph for batched message passing.
   *
   * The resulting block-diagonal CSR is compatible with all sd.gnn() message-passing ops.
   * Use batchVec with sd.segmentMean/Sum/Max for graph-level readout.
   *
   * @param names names May be null. Arrays of names for the output variables.
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
  public SDVariable[] graphDisjointUnion(String[] names, SDVariable[] Xs, SDVariable[] vals,
      SDVariable[] colIdxs, SDVariable... rowPtrs) {
    SDValidation.validateFloatingPoint("graphDisjointUnion", "Xs", Xs);
    Preconditions.checkArgument(Xs.length >= 1, "Xs has incorrect size/length. Expected: Xs.length >= 1, got %s", Xs.length);
    SDValidation.validateFloatingPoint("graphDisjointUnion", "vals", vals);
    Preconditions.checkArgument(vals.length >= 1, "vals has incorrect size/length. Expected: vals.length >= 1, got %s", vals.length);
    SDValidation.validateInteger("graphDisjointUnion", "colIdxs", colIdxs);
    Preconditions.checkArgument(colIdxs.length >= 1, "colIdxs has incorrect size/length. Expected: colIdxs.length >= 1, got %s", colIdxs.length);
    SDValidation.validateInteger("graphDisjointUnion", "rowPtrs", rowPtrs);
    Preconditions.checkArgument(rowPtrs.length >= 1, "rowPtrs has incorrect size/length. Expected: rowPtrs.length >= 1, got %s", rowPtrs.length);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.sparse.GraphDisjointUnion(sd,Xs, vals, colIdxs, rowPtrs).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable holE(SDVariable head, SDVariable relation, SDVariable tail) {
    SDValidation.validateFloatingPoint("holE", "head", head);
    SDValidation.validateFloatingPoint("holE", "relation", relation);
    SDValidation.validateFloatingPoint("holE", "tail", tail);
    SDVariable hExp = sd.expandDims(head, 2);
    SDVariable one = sd.onesLike(hExp);
    SDVariable zero = sd.zerosLike(hExp);
    SDVariable reSel = sd.concat(2, one, zero);
    SDVariable imSel = sd.concat(2, zero, one);
    SDVariable hc = sd.concat(2, hExp, zero);
    SDVariable tc = sd.concat(2, sd.expandDims(tail, 2), zero);
    SDVariable H = sd.signal().dft(hc, 1, false, false);
    SDVariable T = sd.signal().dft(tc, 1, false, false);
    SDVariable hRe = sd.sum(H.mul(reSel), false, 2);
    SDVariable hIm = sd.sum(H.mul(imSel), false, 2);
    SDVariable tRe = sd.sum(T.mul(reSel), false, 2);
    SDVariable tIm = sd.sum(T.mul(imSel), false, 2);
    SDVariable zRe = hRe.mul(tRe).add(hIm.mul(tIm));
    SDVariable zIm = hRe.mul(tIm).sub(hIm.mul(tRe));
    SDVariable zc = sd.concat(2, sd.expandDims(zRe, 2), sd.expandDims(zIm, 2));
    SDVariable ccorr = sd.signal().dft(zc, 1, true, false);
    SDVariable ccorrRe = sd.sum(ccorr.mul(reSel), false, 2);
    SDVariable out = sd.sum(relation.mul(ccorrRe), false, 1);
    return out;
  }

  /**
   * HolE -- Holographic Embeddings (Nickel et al. 2016): scores a triple by the relation's
   * agreement with the circular correlation of head and tail,
   * score = relation . ccorr(head, tail), where ccorr(a,b) = IDFT(conj(DFT(a)) * DFT(b)).
   * Circular correlation gives ComplEx-level expressiveness (asymmetric relations) at O(d log d)
   * via the Fourier domain; here it is composed from the differentiable DFT op.
   *
   * @param name name May be null. Name for the output variable
   * @param head Head entity embeddings [batch, dim] (FLOATING_POINT type)
   * @param relation Relation embeddings [batch, dim] (FLOATING_POINT type)
   * @param tail Tail entity embeddings [batch, dim] (FLOATING_POINT type)
   * @return score Plausibility score [batch] (higher = better) (FLOATING_POINT type)
   */
  public SDVariable holE(String name, SDVariable head, SDVariable relation, SDVariable tail) {
    SDValidation.validateFloatingPoint("holE", "head", head);
    SDValidation.validateFloatingPoint("holE", "relation", relation);
    SDValidation.validateFloatingPoint("holE", "tail", tail);
    SDVariable hExp = sd.expandDims(head, 2);
    SDVariable one = sd.onesLike(hExp);
    SDVariable zero = sd.zerosLike(hExp);
    SDVariable reSel = sd.concat(2, one, zero);
    SDVariable imSel = sd.concat(2, zero, one);
    SDVariable hc = sd.concat(2, hExp, zero);
    SDVariable tc = sd.concat(2, sd.expandDims(tail, 2), zero);
    SDVariable H = sd.signal().dft(hc, 1, false, false);
    SDVariable T = sd.signal().dft(tc, 1, false, false);
    SDVariable hRe = sd.sum(H.mul(reSel), false, 2);
    SDVariable hIm = sd.sum(H.mul(imSel), false, 2);
    SDVariable tRe = sd.sum(T.mul(reSel), false, 2);
    SDVariable tIm = sd.sum(T.mul(imSel), false, 2);
    SDVariable zRe = hRe.mul(tRe).add(hIm.mul(tIm));
    SDVariable zIm = hRe.mul(tIm).sub(hIm.mul(tRe));
    SDVariable zc = sd.concat(2, sd.expandDims(zRe, 2), sd.expandDims(zIm, 2));
    SDVariable ccorr = sd.signal().dft(zc, 1, true, false);
    SDVariable ccorrRe = sd.sum(ccorr.mul(reSel), false, 2);
    SDVariable out = sd.sum(relation.mul(ccorrRe), false, 1);
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Topological Jaccard link-prediction score: S[i,j] = |N(i) ∩ N(j)| / |N(i) ∪ N(j)|, computed
   * as commonNeighbors(i,j) / (deg_i + deg_j - commonNeighbors(i,j)). Unlike feature-vector
   * Jaccard distance, this measures overlap of graph neighborhoods, normalizing for node degree.
   *
   * @param adj Adjacency matrix [n, n] (symmetric for undirected graphs) (FLOATING_POINT type)
   * @return score Topological Jaccard score matrix [n, n] (FLOATING_POINT type)
   */
  public SDVariable jaccardTopology(SDVariable adj) {
    SDValidation.validateFloatingPoint("jaccardTopology", "adj", adj);
    SDVariable cn = sd.mmul(adj, adj);
    SDVariable deg = sd.sum(adj, false, 1);
    SDVariable ones = sd.onesLike(deg);
    SDVariable degI = sd.mmul(sd.reshape(deg, -1, 1), sd.reshape(ones, 1, -1));
    SDVariable degJ = sd.mmul(sd.reshape(ones, -1, 1), sd.reshape(deg, 1, -1));
    SDVariable union = degI.add(degJ).sub(cn);
    SDVariable out = cn.div(union.add(1e-9));
    return out;
  }

  /**
   * Topological Jaccard link-prediction score: S[i,j] = |N(i) ∩ N(j)| / |N(i) ∪ N(j)|, computed
   * as commonNeighbors(i,j) / (deg_i + deg_j - commonNeighbors(i,j)). Unlike feature-vector
   * Jaccard distance, this measures overlap of graph neighborhoods, normalizing for node degree.
   *
   * @param name name May be null. Name for the output variable
   * @param adj Adjacency matrix [n, n] (symmetric for undirected graphs) (FLOATING_POINT type)
   * @return score Topological Jaccard score matrix [n, n] (FLOATING_POINT type)
   */
  public SDVariable jaccardTopology(String name, SDVariable adj) {
    SDValidation.validateFloatingPoint("jaccardTopology", "adj", adj);
    SDVariable cn = sd.mmul(adj, adj);
    SDVariable deg = sd.sum(adj, false, 1);
    SDVariable ones = sd.onesLike(deg);
    SDVariable degI = sd.mmul(sd.reshape(deg, -1, 1), sd.reshape(ones, 1, -1));
    SDVariable degJ = sd.mmul(sd.reshape(ones, -1, 1), sd.reshape(deg, 1, -1));
    SDVariable union = degI.add(degJ).sub(cn);
    SDVariable out = cn.div(union.add(1e-9));
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable katzIndex(SDVariable adj, double beta, int L) {
    SDValidation.validateFloatingPoint("katzIndex", "adj", adj);
    SDVariable S = adj.mul(beta);
    SDVariable Apow = adj;
    double betaPow = beta;
    for (int l = 2; l <= L; l++) {
        Apow = sd.mmul(Apow, adj);
        betaPow = betaPow * beta;
        S = S.add(Apow.mul(betaPow));
    }
    SDVariable out = S;
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
   * @param name name May be null. Name for the output variable
   * @param adj Adjacency matrix [n, n] (FLOATING_POINT type)
   * @param beta Attenuation factor (0 &lt; beta &lt; 1 / spectral_radius(A))
   * @param L Truncation depth: number of path-length terms (>= 1)
   * @return out Katz similarity matrix [n, n] (FLOATING_POINT type)
   */
  public SDVariable katzIndex(String name, SDVariable adj, double beta, int L) {
    SDValidation.validateFloatingPoint("katzIndex", "adj", adj);
    SDVariable S = adj.mul(beta);
    SDVariable Apow = adj;
    double betaPow = beta;
    for (int l = 2; l <= L; l++) {
        Apow = sd.mmul(Apow, adj);
        betaPow = betaPow * beta;
        S = S.add(Apow.mul(betaPow));
    }
    SDVariable out = S;
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable knnGraph(SDVariable similarity, int k, int n) {
    SDValidation.validateFloatingPoint("knnGraph", "similarity", similarity);
    SDVariable[] tk = sd.nn().topK(similarity, k, false);
    SDVariable oneHotIdx = sd.oneHot(tk[1], n, -1, 1.0, 0.0, similarity.dataType());
    SDVariable valsExp = sd.expandDims(tk[0], -1);
    SDVariable out = sd.sum(oneHotIdx.mul(valsExp), false, 1);
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
   * @param name name May be null. Name for the output variable
   * @param similarity Pairwise similarity / affinity matrix [n, n] (FLOATING_POINT type)
   * @param k Number of nearest neighbors to keep per row
   * @param n Number of nodes (matrix dimension); the one-hot depth
   * @return adj kNN adjacency [n, n]: each row keeps its top-k similarities, rest 0 (FLOATING_POINT type)
   */
  public SDVariable knnGraph(String name, SDVariable similarity, int k, int n) {
    SDValidation.validateFloatingPoint("knnGraph", "similarity", similarity);
    SDVariable[] tk = sd.nn().topK(similarity, k, false);
    SDVariable oneHotIdx = sd.oneHot(tk[1], n, -1, 1.0, 0.0, similarity.dataType());
    SDVariable valsExp = sd.expandDims(tk[0], -1);
    SDVariable out = sd.sum(oneHotIdx.mul(valsExp), false, 1);
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable labelPropagation(SDVariable seedY, SDVariable aNormVals, SDVariable aNormColIdx,
      SDVariable aNormRowPtr, int rows, int cols, int k, double alpha) {
    SDValidation.validateFloatingPoint("labelPropagation", "seedY", seedY);
    SDValidation.validateFloatingPoint("labelPropagation", "aNormVals", aNormVals);
    SDValidation.validateInteger("labelPropagation", "aNormColIdx", aNormColIdx);
    SDValidation.validateInteger("labelPropagation", "aNormRowPtr", aNormRowPtr);
    SDVariable Y = seedY;
    for (int i = 0; i < k; i++) {
        SDVariable AY = sd.sparse().csrSpmm(aNormVals, aNormColIdx, aNormRowPtr, Y, rows, cols, false);
        Y = AY.mul(1.0 - alpha).add(seedY.mul(alpha));
    }
    SDVariable out = Y;
    return out;
  }

  /**
   * Personalized-PageRank label propagation (Zhou et al. 2004; APPNP propagation of labels).
   * Diffuses seed/observed label rows over the graph while retaining a fraction alpha of the seed
   * at every step: Y = (1-alpha)*(A_norm . Y) + alpha*seedY. A transductive (semi-supervised) classifier.
   *
   * @param name name May be null. Name for the output variable
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
  public SDVariable labelPropagation(String name, SDVariable seedY, SDVariable aNormVals,
      SDVariable aNormColIdx, SDVariable aNormRowPtr, int rows, int cols, int k, double alpha) {
    SDValidation.validateFloatingPoint("labelPropagation", "seedY", seedY);
    SDValidation.validateFloatingPoint("labelPropagation", "aNormVals", aNormVals);
    SDValidation.validateInteger("labelPropagation", "aNormColIdx", aNormColIdx);
    SDValidation.validateInteger("labelPropagation", "aNormRowPtr", aNormRowPtr);
    SDVariable Y = seedY;
    for (int i = 0; i < k; i++) {
        SDVariable AY = sd.sparse().csrSpmm(aNormVals, aNormColIdx, aNormRowPtr, Y, rows, cols, false);
        Y = AY.mul(1.0 - alpha).add(seedY.mul(alpha));
    }
    SDVariable out = Y;
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable marginRankingLoss(SDVariable posScore, SDVariable negScore, double margin) {
    SDValidation.validateFloatingPoint("marginRankingLoss", "posScore", posScore);
    SDValidation.validateFloatingPoint("marginRankingLoss", "negScore", negScore);
    SDVariable hinge = sd.nn().relu(negScore.sub(posScore).add(margin), 0.0);
    SDVariable out = sd.mean(hinge);
    return out;
  }

  /**
   * Margin ranking loss for KGE training: pushes positive triples to score higher than negatives
   * by at least margin.
   * loss = mean( max(0, margin - posScore + negScore) )
   *
   * @param name name May be null. Name for the output variable
   * @param posScore Scores of true triples [batch] (FLOATING_POINT type)
   * @param negScore Scores of corrupted (negative) triples [batch] (FLOATING_POINT type)
   * @param margin Desired score margin
   * @return loss Scalar margin ranking loss (FLOATING_POINT type)
   */
  public SDVariable marginRankingLoss(String name, SDVariable posScore, SDVariable negScore,
      double margin) {
    SDValidation.validateFloatingPoint("marginRankingLoss", "posScore", posScore);
    SDValidation.validateFloatingPoint("marginRankingLoss", "negScore", negScore);
    SDVariable hinge = sd.nn().relu(negScore.sub(posScore).add(margin), 0.0);
    SDVariable out = sd.mean(hinge);
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable personalizedPageRank(SDVariable aNorm, SDVariable seed, double alpha,
      int iterations) {
    SDValidation.validateFloatingPoint("personalizedPageRank", "aNorm", aNorm);
    SDValidation.validateFloatingPoint("personalizedPageRank", "seed", seed);
    SDVariable r = seed;
    for (int i = 0; i < iterations; i++) {
        r = sd.mmul(aNorm, r).mul(alpha).add(seed.mul(1.0 - alpha));
    }
    SDVariable out = r;
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
   * @param name name May be null. Name for the output variable
   * @param aNorm Row-normalized adjacency [n, n] (FLOATING_POINT type)
   * @param seed Personalization / seed distribution [n, c] (FLOATING_POINT type)
   * @param alpha Propagation weight (teleport = 1-alpha) in (0,1)
   * @param iterations Number of power iterations
   * @return out Personalized PageRank score matrix [n, c] (FLOATING_POINT type)
   */
  public SDVariable personalizedPageRank(String name, SDVariable aNorm, SDVariable seed,
      double alpha, int iterations) {
    SDValidation.validateFloatingPoint("personalizedPageRank", "aNorm", aNorm);
    SDValidation.validateFloatingPoint("personalizedPageRank", "seed", seed);
    SDVariable r = seed;
    for (int i = 0; i < iterations; i++) {
        r = sd.mmul(aNorm, r).mul(alpha).add(seed.mul(1.0 - alpha));
    }
    SDVariable out = r;
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Preferential-Attachment link-prediction score: S[i,j] = deg_i · deg_j (the outer product of
   * the degree vector). Encodes the "rich get richer" hypothesis that high-degree nodes are more
   * likely to acquire new links, independent of any shared neighborhood.
   *
   * @param adj Adjacency matrix [n, n] (FLOATING_POINT type)
   * @return score Preferential-attachment score matrix [n, n] (FLOATING_POINT type)
   */
  public SDVariable preferentialAttachment(SDVariable adj) {
    SDValidation.validateFloatingPoint("preferentialAttachment", "adj", adj);
    SDVariable deg = sd.sum(adj, false, 1);
    SDVariable out = sd.mmul(sd.reshape(deg, -1, 1), sd.reshape(deg, 1, -1));
    return out;
  }

  /**
   * Preferential-Attachment link-prediction score: S[i,j] = deg_i · deg_j (the outer product of
   * the degree vector). Encodes the "rich get richer" hypothesis that high-degree nodes are more
   * likely to acquire new links, independent of any shared neighborhood.
   *
   * @param name name May be null. Name for the output variable
   * @param adj Adjacency matrix [n, n] (FLOATING_POINT type)
   * @return score Preferential-attachment score matrix [n, n] (FLOATING_POINT type)
   */
  public SDVariable preferentialAttachment(String name, SDVariable adj) {
    SDValidation.validateFloatingPoint("preferentialAttachment", "adj", adj);
    SDVariable deg = sd.sum(adj, false, 1);
    SDVariable out = sd.mmul(sd.reshape(deg, -1, 1), sd.reshape(deg, 1, -1));
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Resource-Allocation link-prediction score: S[i,j] = sum_v A[i,v]·A[v,j] / deg_v. Like
   * Adamic-Adar but penalizes high-degree shared neighbors even more strongly (inverse degree
   * rather than inverse log-degree). Often the strongest of the simple topological predictors.
   *
   * @param adj Adjacency matrix [n, n] with positive node degrees (FLOATING_POINT type)
   * @return score Resource-allocation score matrix [n, n] (FLOATING_POINT type)
   */
  public SDVariable resourceAllocation(SDVariable adj) {
    SDValidation.validateFloatingPoint("resourceAllocation", "adj", adj);
    SDVariable deg = sd.sum(adj, false, 1);
    SDVariable scaled = adj.div(sd.reshape(deg, -1, 1));
    SDVariable out = sd.mmul(adj, scaled);
    return out;
  }

  /**
   * Resource-Allocation link-prediction score: S[i,j] = sum_v A[i,v]·A[v,j] / deg_v. Like
   * Adamic-Adar but penalizes high-degree shared neighbors even more strongly (inverse degree
   * rather than inverse log-degree). Often the strongest of the simple topological predictors.
   *
   * @param name name May be null. Name for the output variable
   * @param adj Adjacency matrix [n, n] with positive node degrees (FLOATING_POINT type)
   * @return score Resource-allocation score matrix [n, n] (FLOATING_POINT type)
   */
  public SDVariable resourceAllocation(String name, SDVariable adj) {
    SDValidation.validateFloatingPoint("resourceAllocation", "adj", adj);
    SDVariable deg = sd.sum(adj, false, 1);
    SDVariable scaled = adj.div(sd.reshape(deg, -1, 1));
    SDVariable out = sd.mmul(adj, scaled);
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable rotatE(SDVariable hRe, SDVariable hIm, SDVariable relPhase, SDVariable tRe,
      SDVariable tIm) {
    SDValidation.validateFloatingPoint("rotatE", "hRe", hRe);
    SDValidation.validateFloatingPoint("rotatE", "hIm", hIm);
    SDValidation.validateFloatingPoint("rotatE", "relPhase", relPhase);
    SDValidation.validateFloatingPoint("rotatE", "tRe", tRe);
    SDValidation.validateFloatingPoint("rotatE", "tIm", tIm);
    SDVariable cos = sd.math().cos(relPhase);
    SDVariable sin = sd.math().sin(relPhase);
    SDVariable dRe = hRe.mul(cos).sub(hIm.mul(sin)).sub(tRe);
    SDVariable dIm = hRe.mul(sin).add(hIm.mul(cos)).sub(tIm);
    SDVariable dist = sd.math().sqrt(sd.sum(dRe.mul(dRe).add(dIm.mul(dIm)), false, 1).add(1e-9));
    SDVariable out = sd.math().neg(dist);
    return out;
  }

  /**
   * RotatE (Sun et al. 2019): models each relation as an element-wise rotation in complex space
   * (capturing symmetry, inversion and composition).
   * r = (cos(phase), sin(phase)); score = -||head o r - tail||
   *
   * @param name name May be null. Name for the output variable
   * @param hRe Real part of head embeddings [batch, dim] (FLOATING_POINT type)
   * @param hIm Imag part of head embeddings [batch, dim] (FLOATING_POINT type)
   * @param relPhase Relation rotation phases [batch, dim] (radians) (FLOATING_POINT type)
   * @param tRe Real part of tail embeddings [batch, dim] (FLOATING_POINT type)
   * @param tIm Imag part of tail embeddings [batch, dim] (FLOATING_POINT type)
   * @return score Plausibility score [batch] (higher = better) (FLOATING_POINT type)
   */
  public SDVariable rotatE(String name, SDVariable hRe, SDVariable hIm, SDVariable relPhase,
      SDVariable tRe, SDVariable tIm) {
    SDValidation.validateFloatingPoint("rotatE", "hRe", hRe);
    SDValidation.validateFloatingPoint("rotatE", "hIm", hIm);
    SDValidation.validateFloatingPoint("rotatE", "relPhase", relPhase);
    SDValidation.validateFloatingPoint("rotatE", "tRe", tRe);
    SDValidation.validateFloatingPoint("rotatE", "tIm", tIm);
    SDVariable cos = sd.math().cos(relPhase);
    SDVariable sin = sd.math().sin(relPhase);
    SDVariable dRe = hRe.mul(cos).sub(hIm.mul(sin)).sub(tRe);
    SDVariable dIm = hRe.mul(sin).add(hIm.mul(cos)).sub(tIm);
    SDVariable dist = sd.math().sqrt(sd.sum(dRe.mul(dRe).add(dIm.mul(dIm)), false, 1).add(1e-9));
    SDVariable out = sd.math().neg(dist);
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Segment-max pooling: takes max of node embeddings per graph.
   *
   * @param nodeEmb Node embeddings [sumN, F] (FLOATING_POINT type)
   * @param batchVec Node-to-graph index [sumN] (INT type)
   * @return graphEmb Graph-level embeddings [K, F] (FLOATING_POINT type)
   */
  public SDVariable segmentMaxPool(SDVariable nodeEmb, SDVariable batchVec) {
    SDValidation.validateFloatingPoint("segmentMaxPool", "nodeEmb", nodeEmb);
    SDValidation.validateInteger("segmentMaxPool", "batchVec", batchVec);

                SDVariable out = sd.segmentMax(nodeEmb, batchVec);

    return out;
  }

  /**
   * Segment-max pooling: takes max of node embeddings per graph.
   *
   * @param name name May be null. Name for the output variable
   * @param nodeEmb Node embeddings [sumN, F] (FLOATING_POINT type)
   * @param batchVec Node-to-graph index [sumN] (INT type)
   * @return graphEmb Graph-level embeddings [K, F] (FLOATING_POINT type)
   */
  public SDVariable segmentMaxPool(String name, SDVariable nodeEmb, SDVariable batchVec) {
    SDValidation.validateFloatingPoint("segmentMaxPool", "nodeEmb", nodeEmb);
    SDValidation.validateInteger("segmentMaxPool", "batchVec", batchVec);

                SDVariable out = sd.segmentMax(nodeEmb, batchVec);

    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Segment-mean pooling: produces one embedding per graph from batched node embeddings.
   *
   * @param nodeEmb Node embeddings [sumN, F] (FLOATING_POINT type)
   * @param batchVec Node-to-graph index [sumN] from graphDisjointUnion (INT type)
   * @return graphEmb Graph-level embeddings [K, F] (FLOATING_POINT type)
   */
  public SDVariable segmentMeanPool(SDVariable nodeEmb, SDVariable batchVec) {
    SDValidation.validateFloatingPoint("segmentMeanPool", "nodeEmb", nodeEmb);
    SDValidation.validateInteger("segmentMeanPool", "batchVec", batchVec);

                SDVariable out = sd.segmentMean(nodeEmb, batchVec);

    return out;
  }

  /**
   * Segment-mean pooling: produces one embedding per graph from batched node embeddings.
   *
   * @param name name May be null. Name for the output variable
   * @param nodeEmb Node embeddings [sumN, F] (FLOATING_POINT type)
   * @param batchVec Node-to-graph index [sumN] from graphDisjointUnion (INT type)
   * @return graphEmb Graph-level embeddings [K, F] (FLOATING_POINT type)
   */
  public SDVariable segmentMeanPool(String name, SDVariable nodeEmb, SDVariable batchVec) {
    SDValidation.validateFloatingPoint("segmentMeanPool", "nodeEmb", nodeEmb);
    SDValidation.validateInteger("segmentMeanPool", "batchVec", batchVec);

                SDVariable out = sd.segmentMean(nodeEmb, batchVec);

    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Segment-sum pooling: sums node embeddings per graph.
   *
   * @param nodeEmb Node embeddings [sumN, F] (FLOATING_POINT type)
   * @param batchVec Node-to-graph index [sumN] (INT type)
   * @return graphEmb Graph-level embeddings [K, F] (FLOATING_POINT type)
   */
  public SDVariable segmentSumPool(SDVariable nodeEmb, SDVariable batchVec) {
    SDValidation.validateFloatingPoint("segmentSumPool", "nodeEmb", nodeEmb);
    SDValidation.validateInteger("segmentSumPool", "batchVec", batchVec);

                SDVariable out = sd.segmentSum(nodeEmb, batchVec);

    return out;
  }

  /**
   * Segment-sum pooling: sums node embeddings per graph.
   *
   * @param name name May be null. Name for the output variable
   * @param nodeEmb Node embeddings [sumN, F] (FLOATING_POINT type)
   * @param batchVec Node-to-graph index [sumN] (INT type)
   * @return graphEmb Graph-level embeddings [K, F] (FLOATING_POINT type)
   */
  public SDVariable segmentSumPool(String name, SDVariable nodeEmb, SDVariable batchVec) {
    SDValidation.validateFloatingPoint("segmentSumPool", "nodeEmb", nodeEmb);
    SDValidation.validateInteger("segmentSumPool", "batchVec", batchVec);

                SDVariable out = sd.segmentSum(nodeEmb, batchVec);

    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable set2Set(SDVariable nodeEmb, SDVariable qInit, SDVariable wZr, SDVariable bZr,
      SDVariable wZu, SDVariable bZu, SDVariable wC, SDVariable bC, int processingSteps, long d) {
    SDValidation.validateFloatingPoint("set2Set", "nodeEmb", nodeEmb);
    SDValidation.validateFloatingPoint("set2Set", "qInit", qInit);
    SDValidation.validateFloatingPoint("set2Set", "wZr", wZr);
    SDValidation.validateFloatingPoint("set2Set", "bZr", bZr);
    SDValidation.validateFloatingPoint("set2Set", "wZu", wZu);
    SDValidation.validateFloatingPoint("set2Set", "bZu", bZu);
    SDValidation.validateFloatingPoint("set2Set", "wC", wC);
    SDValidation.validateFloatingPoint("set2Set", "bC", bC);
    SDVariable h = qInit;
    SDVariable xKV = sd.reshape(nodeEmb, 1L, -1L, d);
    SDVariable m = sd.zerosLike(qInit);
    for (int t = 0; t < processingSteps; t++) {
        SDVariable qQ = sd.reshape(h, 1L, 1L, d);
        SDVariable attn3 = sd.nn().dotProductAttentionV2(qQ, xKV, xKV, null, null, 0.0, 0.0, false, false);
        m = sd.reshape(attn3, 1L, d);
        SDVariable xh = sd.concat(1, m, h);
        SDVariable zr = sd.nn().sigmoid(sd.mmul(xh, wZr).add(bZr));
        SDVariable zu = sd.nn().sigmoid(sd.mmul(xh, wZu).add(bZu));
        SDVariable rh = sd.concat(1, m, zr.mul(h));
        SDVariable hh = sd.math().tanh(sd.mmul(rh, wC).add(bC));
        h = h.mul(zu.mul(-1.0).add(1.0)).add(zu.mul(hh));
    }
    SDVariable out = sd.concat(1, m, h);
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
   * @param name name May be null. Name for the output variable
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
  public SDVariable set2Set(String name, SDVariable nodeEmb, SDVariable qInit, SDVariable wZr,
      SDVariable bZr, SDVariable wZu, SDVariable bZu, SDVariable wC, SDVariable bC,
      int processingSteps, long d) {
    SDValidation.validateFloatingPoint("set2Set", "nodeEmb", nodeEmb);
    SDValidation.validateFloatingPoint("set2Set", "qInit", qInit);
    SDValidation.validateFloatingPoint("set2Set", "wZr", wZr);
    SDValidation.validateFloatingPoint("set2Set", "bZr", bZr);
    SDValidation.validateFloatingPoint("set2Set", "wZu", wZu);
    SDValidation.validateFloatingPoint("set2Set", "bZu", bZu);
    SDValidation.validateFloatingPoint("set2Set", "wC", wC);
    SDValidation.validateFloatingPoint("set2Set", "bC", bC);
    SDVariable h = qInit;
    SDVariable xKV = sd.reshape(nodeEmb, 1L, -1L, d);
    SDVariable m = sd.zerosLike(qInit);
    for (int t = 0; t < processingSteps; t++) {
        SDVariable qQ = sd.reshape(h, 1L, 1L, d);
        SDVariable attn3 = sd.nn().dotProductAttentionV2(qQ, xKV, xKV, null, null, 0.0, 0.0, false, false);
        m = sd.reshape(attn3, 1L, d);
        SDVariable xh = sd.concat(1, m, h);
        SDVariable zr = sd.nn().sigmoid(sd.mmul(xh, wZr).add(bZr));
        SDVariable zu = sd.nn().sigmoid(sd.mmul(xh, wZu).add(bZu));
        SDVariable rh = sd.concat(1, m, zr.mul(h));
        SDVariable hh = sd.math().tanh(sd.mmul(rh, wC).add(bC));
        h = h.mul(zu.mul(-1.0).add(1.0)).add(zu.mul(hh));
    }
    SDVariable out = sd.concat(1, m, h);
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable simRank(SDVariable W, SDVariable identity, double C, int iterations) {
    SDValidation.validateFloatingPoint("simRank", "W", W);
    SDValidation.validateFloatingPoint("simRank", "identity", identity);
    SDVariable onesM = sd.onesLike(W);
    SDVariable offDiagMask = onesM.sub(identity);
    SDVariable S = identity;
    for (int i = 0; i < iterations; i++) {
        SDVariable prop = sd.mmul(sd.mmul(W, S, true, false, false), W).mul(C);
        S = prop.mul(offDiagMask).add(identity);
    }
    SDVariable out = S;
    return out;
  }

  /**
   * SimRank node-similarity (Jeh and Widom 2002): two nodes are similar if their in-neighbors are
   * similar. Converges via the fixed-point iteration S_{t+1} = C * W^T S_t W (diagonal forced to 1),
   * where W is the column-normalized adjacency. The diagonal reset is implemented in a gradient-clean
   * elementwise form: S_new = prop * (ones - I) + I.
   * Pass identity = sd.constant(Nd4j.eye(n).castTo(DataType.DOUBLE)) so its gradient is not tracked.
   *
   * @param name name May be null. Name for the output variable
   * @param W Column-normalized adjacency [n, n] (each column sums to 1) (FLOATING_POINT type)
   * @param identity Identity matrix [n, n] -- pass sd.constant(Nd4j.eye(n)) (FLOATING_POINT type)
   * @param C SimRank decay constant in (0, 1)
   * @param iterations Number of power iterations
   * @return out Node-similarity matrix [n, n]; diagonal = 1 (FLOATING_POINT type)
   */
  public SDVariable simRank(String name, SDVariable W, SDVariable identity, double C,
      int iterations) {
    SDValidation.validateFloatingPoint("simRank", "W", W);
    SDValidation.validateFloatingPoint("simRank", "identity", identity);
    SDVariable onesM = sd.onesLike(W);
    SDVariable offDiagMask = onesM.sub(identity);
    SDVariable S = identity;
    for (int i = 0; i < iterations; i++) {
        SDVariable prop = sd.mmul(sd.mmul(W, S, true, false, false), W).mul(C);
        S = prop.mul(offDiagMask).add(identity);
    }
    SDVariable out = S;
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable sortPool(SDVariable features, SDVariable sortKey, int k, int n) {
    SDValidation.validateFloatingPoint("sortPool", "features", features);
    SDValidation.validateFloatingPoint("sortPool", "sortKey", sortKey);
    SDVariable[] tk = sd.nn().topK(sortKey, k, true);
    SDVariable sel = sd.oneHot(tk[1], n, -1, 1.0, 0.0, features.dataType());
    SDVariable out = sd.mmul(sel, features);
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
   * @param name name May be null. Name for the output variable
   * @param features Node feature matrix [n, d] (FLOATING_POINT type)
   * @param sortKey Per-node sort scores [n] (e.g. last GNN channel) (FLOATING_POINT type)
   * @param k Number of nodes to keep (output rows)
   * @param n Number of nodes (one-hot depth)
   * @return pooled Top-k node features in descending sort-key order [k, d] (FLOATING_POINT type)
   */
  public SDVariable sortPool(String name, SDVariable features, SDVariable sortKey, int k, int n) {
    SDValidation.validateFloatingPoint("sortPool", "features", features);
    SDValidation.validateFloatingPoint("sortPool", "sortKey", sortKey);
    SDVariable[] tk = sd.nn().topK(sortKey, k, true);
    SDVariable sel = sd.oneHot(tk[1], n, -1, 1.0, 0.0, features.dataType());
    SDVariable out = sd.mmul(sel, features);
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable topKPool(SDVariable scores, SDVariable features, int k, int n) {
    SDValidation.validateFloatingPoint("topKPool", "scores", scores);
    SDValidation.validateFloatingPoint("topKPool", "features", features);
    SDVariable[] tk = sd.nn().topK(scores, k, false);
    SDVariable sel = sd.oneHot(tk[1], n, -1, 1.0, 0.0, features.dataType());
    SDVariable gathered = sd.mmul(sel, features);
    SDVariable gate = sd.nn().sigmoid(tk[0]);
    SDVariable out = gathered.mul(sd.expandDims(gate, -1));
    return out;
  }

  /**
   * Top-k node pooling (Gao and Ji 2019 / Cangea et al. 2018; the selection mechanism of SAGPool):
   * keeps the k highest-scoring nodes and gates their features by sigmoid(score) so the score stays
   * differentiable. The top-k rows are gathered as oneHot(topIndices, n) @ features (avoiding a
   * separate gather op). Pass scores from a learned projection or a graph-attention layer for SAGPool.
   *
   * @param name name May be null. Name for the output variable
   * @param scores Per-node selection scores [n] (e.g. a learned projection) (FLOATING_POINT type)
   * @param features Node features [n, d] (FLOATING_POINT type)
   * @param k Number of nodes to keep
   * @param n Number of nodes (the one-hot depth)
   * @return pooled Pooled features of the top-k nodes [k, d], gated by sigmoid(score) (FLOATING_POINT type)
   */
  public SDVariable topKPool(String name, SDVariable scores, SDVariable features, int k, int n) {
    SDValidation.validateFloatingPoint("topKPool", "scores", scores);
    SDValidation.validateFloatingPoint("topKPool", "features", features);
    SDVariable[] tk = sd.nn().topK(scores, k, false);
    SDVariable sel = sd.oneHot(tk[1], n, -1, 1.0, 0.0, features.dataType());
    SDVariable gathered = sd.mmul(sel, features);
    SDVariable gate = sd.nn().sigmoid(tk[0]);
    SDVariable out = gathered.mul(sd.expandDims(gate, -1));
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable transE(SDVariable head, SDVariable relation, SDVariable tail) {
    SDValidation.validateFloatingPoint("transE", "head", head);
    SDValidation.validateFloatingPoint("transE", "relation", relation);
    SDValidation.validateFloatingPoint("transE", "tail", tail);
    SDVariable diff = head.add(relation).sub(tail);
    SDVariable dist = sd.math().sqrt(sd.sum(diff.mul(diff), false, 1).add(1e-9));
    SDVariable out = sd.math().neg(dist);
    return out;
  }

  /**
   * TransE (Bordes et al. 2013): models a relation as a translation, h + r ~ t.
   * score = -||head + relation - tail||
   *
   * @param name name May be null. Name for the output variable
   * @param head Head-entity embeddings [batch, dim] (FLOATING_POINT type)
   * @param relation Relation embeddings [batch, dim] (FLOATING_POINT type)
   * @param tail Tail-entity embeddings [batch, dim] (FLOATING_POINT type)
   * @return score Plausibility score [batch] (higher = better) (FLOATING_POINT type)
   */
  public SDVariable transE(String name, SDVariable head, SDVariable relation, SDVariable tail) {
    SDValidation.validateFloatingPoint("transE", "head", head);
    SDValidation.validateFloatingPoint("transE", "relation", relation);
    SDValidation.validateFloatingPoint("transE", "tail", tail);
    SDVariable diff = head.add(relation).sub(tail);
    SDVariable dist = sd.math().sqrt(sd.sum(diff.mul(diff), false, 1).add(1e-9));
    SDVariable out = sd.math().neg(dist);
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable transET(SDVariable head, SDVariable relation, SDVariable time,
      SDVariable tail) {
    SDValidation.validateFloatingPoint("transET", "head", head);
    SDValidation.validateFloatingPoint("transET", "relation", relation);
    SDValidation.validateFloatingPoint("transET", "time", time);
    SDValidation.validateFloatingPoint("transET", "tail", tail);
    SDVariable diff = head.add(relation).add(time).sub(tail);
    SDVariable dist = sd.math().sqrt(sd.sum(diff.mul(diff), false, 1).add(1e-9));
    SDVariable out = sd.math().neg(dist);
    return out;
  }

  /**
   * Time-aware TransE (TTransE, Jiang et al. 2016) for temporal knowledge graphs: the timestamp
   * embedding is an additional translation.
   * score = -||head + relation + time - tail||
   *
   * @param name name May be null. Name for the output variable
   * @param head Head-entity embeddings [batch, dim] (FLOATING_POINT type)
   * @param relation Relation embeddings [batch, dim] (FLOATING_POINT type)
   * @param time Timestamp embeddings [batch, dim] (FLOATING_POINT type)
   * @param tail Tail-entity embeddings [batch, dim] (FLOATING_POINT type)
   * @return score Plausibility score [batch] (higher = better) (FLOATING_POINT type)
   */
  public SDVariable transET(String name, SDVariable head, SDVariable relation, SDVariable time,
      SDVariable tail) {
    SDValidation.validateFloatingPoint("transET", "head", head);
    SDValidation.validateFloatingPoint("transET", "relation", relation);
    SDValidation.validateFloatingPoint("transET", "time", time);
    SDValidation.validateFloatingPoint("transET", "tail", tail);
    SDVariable diff = head.add(relation).add(time).sub(tail);
    SDVariable dist = sd.math().sqrt(sd.sum(diff.mul(diff), false, 1).add(1e-9));
    SDVariable out = sd.math().neg(dist);
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable transH(SDVariable head, SDVariable wr, SDVariable relation, SDVariable tail) {
    SDValidation.validateFloatingPoint("transH", "head", head);
    SDValidation.validateFloatingPoint("transH", "wr", wr);
    SDValidation.validateFloatingPoint("transH", "relation", relation);
    SDValidation.validateFloatingPoint("transH", "tail", tail);
    SDVariable hPerp = head.sub(wr.mul(sd.sum(wr.mul(head), true, 1)));
    SDVariable tPerp = tail.sub(wr.mul(sd.sum(wr.mul(tail), true, 1)));
    SDVariable diff = hPerp.add(relation).sub(tPerp);
    SDVariable dist = sd.math().sqrt(sd.sum(diff.mul(diff), false, 1).add(1e-9));
    SDVariable out = sd.math().neg(dist);
    return out;
  }

  /**
   * TransH (Wang et al. 2014): like TransE, but head and tail are projected onto a relation-specific
   * hyperplane (normal wr), so an entity can play different roles under different relations.
   * score = -||hPerp + relation - tPerp||
   *
   * @param name name May be null. Name for the output variable
   * @param head Head-entity embeddings [batch, dim] (FLOATING_POINT type)
   * @param wr Relation hyperplane normals [batch, dim] (ideally unit-norm) (FLOATING_POINT type)
   * @param relation Relation translation embeddings [batch, dim] (FLOATING_POINT type)
   * @param tail Tail-entity embeddings [batch, dim] (FLOATING_POINT type)
   * @return score Plausibility score [batch] (higher = better) (FLOATING_POINT type)
   */
  public SDVariable transH(String name, SDVariable head, SDVariable wr, SDVariable relation,
      SDVariable tail) {
    SDValidation.validateFloatingPoint("transH", "head", head);
    SDValidation.validateFloatingPoint("transH", "wr", wr);
    SDValidation.validateFloatingPoint("transH", "relation", relation);
    SDValidation.validateFloatingPoint("transH", "tail", tail);
    SDVariable hPerp = head.sub(wr.mul(sd.sum(wr.mul(head), true, 1)));
    SDVariable tPerp = tail.sub(wr.mul(sd.sum(wr.mul(tail), true, 1)));
    SDVariable diff = hPerp.add(relation).sub(tPerp);
    SDVariable dist = sd.math().sqrt(sd.sum(diff.mul(diff), false, 1).add(1e-9));
    SDVariable out = sd.math().neg(dist);
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable tuckER(SDVariable head, SDVariable relation, SDVariable tail, SDVariable coreW,
      int de, int dr) {
    SDValidation.validateFloatingPoint("tuckER", "head", head);
    SDValidation.validateFloatingPoint("tuckER", "relation", relation);
    SDValidation.validateFloatingPoint("tuckER", "tail", tail);
    SDValidation.validateFloatingPoint("tuckER", "coreW", coreW);
    SDVariable coreUnfold = sd.reshape(coreW, de, dr * de);
    SDVariable m1 = sd.reshape(sd.mmul(head, coreUnfold), -1, dr, de);
    SDVariable rExp = sd.reshape(relation, -1, dr, 1);
    SDVariable m2 = sd.sum(rExp.mul(m1), false, 1);
    SDVariable out = sd.sum(m2.mul(tail), false, 1);
    return out;
  }

  /**
   * TuckER (Balazevic et al. 2019): a Tucker-decomposition bilinear model with a learnable core
   * tensor shared across all triples; subsumes DistMult / ComplEx / SimplE.
   * score = W x1 head x2 relation x3 tail
   *
   * @param name name May be null. Name for the output variable
   * @param head Head-entity embeddings [batch, de] (FLOATING_POINT type)
   * @param relation Relation embeddings [batch, dr] (FLOATING_POINT type)
   * @param tail Tail-entity embeddings [batch, de] (FLOATING_POINT type)
   * @param coreW Core tensor [de, dr, de] (FLOATING_POINT type)
   * @param de Entity embedding dimension
   * @param dr Relation embedding dimension
   * @return score Plausibility score [batch] (higher = better) (FLOATING_POINT type)
   */
  public SDVariable tuckER(String name, SDVariable head, SDVariable relation, SDVariable tail,
      SDVariable coreW, int de, int dr) {
    SDValidation.validateFloatingPoint("tuckER", "head", head);
    SDValidation.validateFloatingPoint("tuckER", "relation", relation);
    SDValidation.validateFloatingPoint("tuckER", "tail", tail);
    SDValidation.validateFloatingPoint("tuckER", "coreW", coreW);
    SDVariable coreUnfold = sd.reshape(coreW, de, dr * de);
    SDVariable m1 = sd.reshape(sd.mmul(head, coreUnfold), -1, dr, de);
    SDVariable rExp = sd.reshape(relation, -1, dr, 1);
    SDVariable m2 = sd.sum(rExp.mul(m1), false, 1);
    SDVariable out = sd.sum(m2.mul(tail), false, 1);
    return sd.updateVariableNameAndReference(out, name);
  }
}
