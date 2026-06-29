/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.codegen.ops

import org.nd4j.codegen.api.DataType.*
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*

/**
 * Graph-analysis namespace: knowledge-graph embedding (KGE) scorers together with graph-construction
 * and graph-learning operations. Every method is a pure SameDiff composition declared via
 * [Composition]; the generator emits both the SameDiff (sd.graph()) form and the eager (Nd4j.graph())
 * form from a single definition. The differentiable message-passing layers live in the GNN namespace
 * (sd.gnn()).
 */
fun Graph() = Namespace("Graph") {

    // -------------------------------------------------------------------------
    // Knowledge-graph embedding scorers (triple (head, relation, tail) plausibility)
    // -------------------------------------------------------------------------

    Op("transE") {
        Input(FLOATING_POINT, "head")     { description = "Head-entity embeddings [batch, dim]" }
        Input(FLOATING_POINT, "relation") { description = "Relation embeddings [batch, dim]" }
        Input(FLOATING_POINT, "tail")     { description = "Tail-entity embeddings [batch, dim]" }
        Output(FLOATING_POINT, "score")   { description = "Plausibility score [batch] (higher = better)" }
        Composition("""
            SDVariable diff = head.add(relation).sub(tail);
            SDVariable dist = sd.math().sqrt(sd.sum(diff.mul(diff), false, 1).add(1e-9));
            SDVariable out = sd.math().neg(dist);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            TransE (Bordes et al. 2013): models a relation as a translation, h + r ~ t.
            score = -||head + relation - tail||
            """.trimIndent()
        }
    }

    Op("distMult") {
        Input(FLOATING_POINT, "head")     { description = "Head-entity embeddings [batch, dim]" }
        Input(FLOATING_POINT, "relation") { description = "Relation embeddings [batch, dim]" }
        Input(FLOATING_POINT, "tail")     { description = "Tail-entity embeddings [batch, dim]" }
        Output(FLOATING_POINT, "score")   { description = "Plausibility score [batch] (higher = better)" }
        Composition("""
            SDVariable out = sd.sum(head.mul(relation).mul(tail), false, 1);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            DistMult (Yang et al. 2015): a symmetric trilinear product.
            score = sum_d head_d * relation_d * tail_d
            """.trimIndent()
        }
    }

    Op("complEx") {
        Input(FLOATING_POINT, "hRe") { description = "Real part of head embeddings [batch, dim]" }
        Input(FLOATING_POINT, "hIm") { description = "Imag part of head embeddings [batch, dim]" }
        Input(FLOATING_POINT, "rRe") { description = "Real part of relation embeddings [batch, dim]" }
        Input(FLOATING_POINT, "rIm") { description = "Imag part of relation embeddings [batch, dim]" }
        Input(FLOATING_POINT, "tRe") { description = "Real part of tail embeddings [batch, dim]" }
        Input(FLOATING_POINT, "tIm") { description = "Imag part of tail embeddings [batch, dim]" }
        Output(FLOATING_POINT, "score") { description = "Plausibility score [batch] (higher = better)" }
        Composition("""
            SDVariable s = hRe.mul(rRe).mul(tRe).add(hRe.mul(rIm).mul(tIm)).add(hIm.mul(rRe).mul(tIm)).sub(hIm.mul(rIm).mul(tRe));
            SDVariable out = sd.sum(s, false, 1);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            ComplEx (Trouillon et al. 2016): a complex-valued bilinear product whose real part scores the
            triple; the imaginary parts let it model asymmetric relations.
            score = Re( sum_d head_d * relation_d * conj(tail_d) )
            """.trimIndent()
        }
    }

    Op("rotatE") {
        Input(FLOATING_POINT, "hRe")      { description = "Real part of head embeddings [batch, dim]" }
        Input(FLOATING_POINT, "hIm")      { description = "Imag part of head embeddings [batch, dim]" }
        Input(FLOATING_POINT, "relPhase") { description = "Relation rotation phases [batch, dim] (radians)" }
        Input(FLOATING_POINT, "tRe")      { description = "Real part of tail embeddings [batch, dim]" }
        Input(FLOATING_POINT, "tIm")      { description = "Imag part of tail embeddings [batch, dim]" }
        Output(FLOATING_POINT, "score")   { description = "Plausibility score [batch] (higher = better)" }
        Composition("""
            SDVariable cos = sd.math().cos(relPhase);
            SDVariable sin = sd.math().sin(relPhase);
            SDVariable dRe = hRe.mul(cos).sub(hIm.mul(sin)).sub(tRe);
            SDVariable dIm = hRe.mul(sin).add(hIm.mul(cos)).sub(tIm);
            SDVariable dist = sd.math().sqrt(sd.sum(dRe.mul(dRe).add(dIm.mul(dIm)), false, 1).add(1e-9));
            SDVariable out = sd.math().neg(dist);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            RotatE (Sun et al. 2019): models each relation as an element-wise rotation in complex space
            (capturing symmetry, inversion and composition).
            r = (cos(phase), sin(phase)); score = -||head o r - tail||
            """.trimIndent()
        }
    }

    Op("marginRankingLoss") {
        Input(FLOATING_POINT, "posScore") { description = "Scores of true triples [batch]" }
        Input(FLOATING_POINT, "negScore") { description = "Scores of corrupted (negative) triples [batch]" }
        Arg(FLOATING_POINT, "margin")     { description = "Desired score margin" }
        Output(FLOATING_POINT, "loss")    { description = "Scalar margin ranking loss" }
        Composition("""
            SDVariable hinge = sd.nn().relu(negScore.sub(posScore).add(margin), 0.0);
            SDVariable out = sd.mean(hinge);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Margin ranking loss for KGE training: pushes positive triples to score higher than negatives
            by at least margin.
            loss = mean( max(0, margin - posScore + negScore) )
            """.trimIndent()
        }
    }

    Op("transH") {
        Input(FLOATING_POINT, "head")     { description = "Head-entity embeddings [batch, dim]" }
        Input(FLOATING_POINT, "wr")       { description = "Relation hyperplane normals [batch, dim] (ideally unit-norm)" }
        Input(FLOATING_POINT, "relation") { description = "Relation translation embeddings [batch, dim]" }
        Input(FLOATING_POINT, "tail")     { description = "Tail-entity embeddings [batch, dim]" }
        Output(FLOATING_POINT, "score")   { description = "Plausibility score [batch] (higher = better)" }
        Composition("""
            SDVariable hPerp = head.sub(wr.mul(sd.sum(wr.mul(head), true, 1)));
            SDVariable tPerp = tail.sub(wr.mul(sd.sum(wr.mul(tail), true, 1)));
            SDVariable diff = hPerp.add(relation).sub(tPerp);
            SDVariable dist = sd.math().sqrt(sd.sum(diff.mul(diff), false, 1).add(1e-9));
            SDVariable out = sd.math().neg(dist);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            TransH (Wang et al. 2014): like TransE, but head and tail are projected onto a relation-specific
            hyperplane (normal wr), so an entity can play different roles under different relations.
            score = -||hPerp + relation - tPerp||
            """.trimIndent()
        }
    }

    Op("transET") {
        Input(FLOATING_POINT, "head")     { description = "Head-entity embeddings [batch, dim]" }
        Input(FLOATING_POINT, "relation") { description = "Relation embeddings [batch, dim]" }
        Input(FLOATING_POINT, "time")     { description = "Timestamp embeddings [batch, dim]" }
        Input(FLOATING_POINT, "tail")     { description = "Tail-entity embeddings [batch, dim]" }
        Output(FLOATING_POINT, "score")   { description = "Plausibility score [batch] (higher = better)" }
        Composition("""
            SDVariable diff = head.add(relation).add(time).sub(tail);
            SDVariable dist = sd.math().sqrt(sd.sum(diff.mul(diff), false, 1).add(1e-9));
            SDVariable out = sd.math().neg(dist);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Time-aware TransE (TTransE, Jiang et al. 2016) for temporal knowledge graphs: the timestamp
            embedding is an additional translation.
            score = -||head + relation + time - tail||
            """.trimIndent()
        }
    }

    Op("tuckER") {
        Input(FLOATING_POINT, "head")     { description = "Head-entity embeddings [batch, de]" }
        Input(FLOATING_POINT, "relation") { description = "Relation embeddings [batch, dr]" }
        Input(FLOATING_POINT, "tail")     { description = "Tail-entity embeddings [batch, de]" }
        Input(FLOATING_POINT, "coreW")    { description = "Core tensor [de, dr, de]" }
        Arg(INT, "de")                    { description = "Entity embedding dimension" }
        Arg(INT, "dr")                    { description = "Relation embedding dimension" }
        Output(FLOATING_POINT, "score")   { description = "Plausibility score [batch] (higher = better)" }
        Composition("""
            SDVariable coreUnfold = sd.reshape(coreW, de, dr * de);
            SDVariable m1 = sd.reshape(sd.mmul(head, coreUnfold), -1, dr, de);
            SDVariable rExp = sd.reshape(relation, -1, dr, 1);
            SDVariable m2 = sd.sum(rExp.mul(m1), false, 1);
            SDVariable out = sd.sum(m2.mul(tail), false, 1);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            TuckER (Balazevic et al. 2019): a Tucker-decomposition bilinear model with a learnable core
            tensor shared across all triples; subsumes DistMult / ComplEx / SimplE.
            score = W x1 head x2 relation x3 tail
            """.trimIndent()
        }
    }

    Op("convE") {
        Input(FLOATING_POINT, "head")     { description = "Head-entity embeddings [batch, de] (de = embH*embW)" }
        Input(FLOATING_POINT, "relation") { description = "Relation embeddings [batch, de]" }
        Input(FLOATING_POINT, "tail")     { description = "Tail-entity embeddings [batch, de]" }
        Input(FLOATING_POINT, "convW")    { description = "Conv weights [3, 3, 1, channels]" }
        Input(FLOATING_POINT, "convB")    { description = "Conv bias [channels]" }
        Input(FLOATING_POINT, "fcW")      { description = "FC weights [channels*(2*embH-2)*(embW-2), de]" }
        Input(FLOATING_POINT, "fcB")      { description = "FC bias [de]" }
        Arg(INT, "embH")                  { description = "Reshape height per embedding" }
        Arg(INT, "embW")                  { description = "Reshape width (de = embH*embW)" }
        Arg(INT, "channels")              { description = "Conv output channels" }
        Output(FLOATING_POINT, "score")   { description = "Plausibility score [batch] (higher = better)" }
        Composition("""
            SDVariable hImg = sd.reshape(head, -1, embH, embW);
            SDVariable rImg = sd.reshape(relation, -1, embH, embW);
            SDVariable stacked = sd.concat(1, hImg, rImg);
            SDVariable img = sd.reshape(stacked, -1, 1, 2L * embH, embW);
            org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig cfg = org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig.builder().kH(3).kW(3).build();
            SDVariable feat = sd.nn().relu(sd.cnn().conv2d(img, convW, convB, cfg), 0.0);
            long flatDim = (long) channels * (2L * embH - 2) * (embW - 2);
            SDVariable flat = sd.reshape(feat, -1, flatDim);
            SDVariable fc = sd.nn().relu(sd.mmul(flat, fcW).add(fcB), 0.0);
            SDVariable out = sd.sum(fc.mul(tail), false, 1);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            ConvE (Dettmers et al. 2018): reshape head + relation into 2D images, stack, run a 2D
            convolution + fully-connected projection, then score against the tail. A strong,
            parameter-efficient KGE baseline.
            """.trimIndent()
        }
    }

    Op("holE") {
        Input(FLOATING_POINT, "head")     { description = "Head entity embeddings [batch, dim]" }
        Input(FLOATING_POINT, "relation") { description = "Relation embeddings [batch, dim]" }
        Input(FLOATING_POINT, "tail")     { description = "Tail entity embeddings [batch, dim]" }
        Output(FLOATING_POINT, "score")   { description = "Plausibility score [batch] (higher = better)" }
        Composition("""
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
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            HolE -- Holographic Embeddings (Nickel et al. 2016): scores a triple by the relation's
            agreement with the circular correlation of head and tail,
            score = relation . ccorr(head, tail), where ccorr(a,b) = IDFT(conj(DFT(a)) * DFT(b)).
            Circular correlation gives ComplEx-level expressiveness (asymmetric relations) at O(d log d)
            via the Fourier domain; here it is composed from the differentiable DFT op.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // Graph construction from data
    // -------------------------------------------------------------------------

    Op("correlationMatrix") {
        Input(FLOATING_POINT, "data")   { description = "Observations x variables [n, d]" }
        Output(FLOATING_POINT, "corr")  { description = "Pearson correlation matrix [d, d] with unit diagonal" }
        Composition("""
            SDVariable mean = sd.mean(data, true, 0);
            SDVariable centered = data.sub(mean);
            SDVariable cov = sd.mmul(centered, centered, true, false, false);
            SDVariable variance = sd.sum(centered.mul(centered), false, 0);
            SDVariable std = sd.math().sqrt(variance.add(1e-12));
            SDVariable denom = sd.mmul(sd.reshape(std, -1, 1), sd.reshape(std, 1, -1));
            SDVariable out = cov.div(denom);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Pearson correlation matrix among the columns (variables) of a data matrix -- the basis of a
            correlation graph. Threshold |corr| with sd.sparse().denseToCsr(...) to obtain a sparse
            correlation-graph adjacency.
            """.trimIndent()
        }
    }

    Op("cosineSimilarity") {
        Input(FLOATING_POINT, "features") { description = "Node features [n, d]" }
        Output(FLOATING_POINT, "sim")     { description = "Cosine-similarity matrix [n, n] with unit diagonal" }
        Composition("""
            SDVariable norm = sd.math().sqrt(sd.sum(features.mul(features), true, 1).add(1e-12));
            SDVariable normalized = features.div(norm);
            SDVariable out = sd.mmul(normalized, normalized, false, true, false);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Cosine-similarity affinity between the rows (nodes) of a feature matrix.
            sim[i,j] = (x_i . x_j) / (||x_i|| ||x_j||). A standard input to a kNN / thresholded similarity graph.
            """.trimIndent()
        }
    }

    Op("gaussianSimilarity") {
        Input(FLOATING_POINT, "features") { description = "Node features [n, d]" }
        Arg(FLOATING_POINT, "sigma")      { description = "RBF kernel bandwidth (> 0)" }
        Output(FLOATING_POINT, "sim")     { description = "Gaussian-similarity matrix [n, n] with unit diagonal" }
        Composition("""
            SDVariable xi = sd.expandDims(features, 1);
            SDVariable xj = sd.expandDims(features, 0);
            SDVariable diff = xi.sub(xj);
            SDVariable d2 = sd.sum(diff.mul(diff), false, 2);
            SDVariable out = sd.math().exp(d2.mul(-1.0 / (2.0 * sigma * sigma)));
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Gaussian (RBF) similarity affinity between the rows (nodes) of a feature matrix.
            sim[i,j] = exp( -||x_i - x_j||^2 / (2 sigma^2) ). Threshold or take per-row top-k for a kNN graph.
            """.trimIndent()
        }
    }

    Op("knnGraph") {
        Input(FLOATING_POINT, "similarity") { description = "Pairwise similarity / affinity matrix [n, n]" }
        Arg(INT, "k")                       { description = "Number of nearest neighbors to keep per row" }
        Arg(INT, "n")                       { description = "Number of nodes (matrix dimension); the one-hot depth" }
        Output(FLOATING_POINT, "adj")       { description = "kNN adjacency [n, n]: each row keeps its top-k similarities, rest 0" }
        Composition("""
            SDVariable[] tk = sd.nn().topK(similarity, k, false);
            SDVariable oneHotIdx = sd.oneHot(tk[1], n, -1, 1.0, 0.0, similarity.dataType());
            SDVariable valsExp = sd.expandDims(tk[0], -1);
            SDVariable out = sd.sum(oneHotIdx.mul(valsExp), false, 1);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            k-nearest-neighbor graph construction from a similarity/affinity matrix: each node (row) keeps
            only its k highest-similarity neighbors, zeroing the rest, yielding a sparse weighted adjacency.
            Built by scattering the per-row top-k values back to their column positions
            (sum over k of oneHot(topIndices, n) * topValues), so it is differentiable w.r.t. the kept
            similarities through the TopK gradient. Pair with sd.graph().cosineSimilarity / gaussianSimilarity
            / correlationMatrix to go from raw features straight to a learnable kNN graph. `n` is the node
            count (matrix dimension), used as the one-hot depth.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // Graph learning: transductive classification + self-supervision
    // -------------------------------------------------------------------------

    Op("labelPropagation") {
        Input(FLOATING_POINT, "seedY")       { description = "Seed label distribution per node [n, c]" }
        Input(FLOATING_POINT, "aNormVals")   { description = "CSR values of the normalized adjacency [nnz]" }
        Input(INT, "aNormColIdx")            { description = "CSR column indices [nnz] (INT32)" }
        Input(INT, "aNormRowPtr")            { description = "CSR row pointers [n+1] (INT32)" }
        Arg(INT, "rows")                     { description = "Number of nodes" }
        Arg(INT, "cols")                     { description = "Columns of A_norm (= rows for square graphs)" }
        Arg(INT, "k")                        { description = "Number of propagation steps" }
        Arg(FLOATING_POINT, "alpha")         { description = "Teleport / restart probability in [0,1]" }
        Output(FLOATING_POINT, "labels")     { description = "Propagated label distribution [n, c]" }
        Composition("""
            SDVariable Y = seedY;
            for (int i = 0; i < k; i++) {
                SDVariable AY = sd.sparse().csrSpmm(aNormVals, aNormColIdx, aNormRowPtr, Y, rows, cols, false);
                Y = AY.mul(1.0 - alpha).add(seedY.mul(alpha));
            }
            SDVariable out = Y;
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Personalized-PageRank label propagation (Zhou et al. 2004; APPNP propagation of labels).
            Diffuses seed/observed label rows over the graph while retaining a fraction alpha of the seed
            at every step: Y = (1-alpha)*(A_norm . Y) + alpha*seedY. A transductive (semi-supervised) classifier.
            """.trimIndent()
        }
    }

    Op("dgiLoss") {
        Input(FLOATING_POINT, "H")     { description = "Node encodings of the real graph [n, d]" }
        Input(FLOATING_POINT, "Hneg")  { description = "Node encodings of the corrupted graph [n, d]" }
        Input(FLOATING_POINT, "discW") { description = "Bilinear discriminator weight [d, d]" }
        Output(FLOATING_POINT, "loss") { description = "Scalar DGI loss" }
        Composition("""
            SDVariable s = sd.nn().sigmoid(sd.mean(H, true, 0));
            SDVariable sW = sd.mmul(s, discW, false, true, false);
            SDVariable pos = sd.sum(H.mul(sW), false, 1);
            SDVariable neg = sd.sum(Hneg.mul(sW), false, 1);
            SDVariable posTerm = sd.math().log(sd.nn().sigmoid(pos).add(1e-12));
            SDVariable negTerm = sd.math().log(sd.nn().sigmoid(neg.mul(-1.0)).add(1e-12));
            SDVariable out = sd.mean(posTerm.add(negTerm), false).mul(-1.0);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Deep Graph Infomax loss (Velickovic et al. 2019): maximizes mutual information between each
            node's encoding and a global graph summary via a bilinear discriminator that tells real node
            encodings from encodings of a corrupted graph. Yields label-free node embeddings.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // Link-prediction heuristics (topology-based edge scores over an adjacency matrix)
    // -------------------------------------------------------------------------

    Op("commonNeighbors") {
        Input(FLOATING_POINT, "adj")    { description = "Adjacency matrix [n, n] (symmetric for undirected graphs)" }
        Output(FLOATING_POINT, "score") { description = "Common-neighbor score matrix [n, n]" }
        Composition("""
            SDVariable out = sd.mmul(adj, adj);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Common-Neighbors link-prediction score: S = A·A, so S[i,j] counts the (weighted) paths of
            length two between i and j -- the number of neighbors they share. The simplest topological
            link predictor; higher scores indicate more likely missing edges.
            """.trimIndent()
        }
    }

    Op("adamicAdar") {
        Input(FLOATING_POINT, "adj")    { description = "Adjacency matrix [n, n] with node degrees > 1" }
        Output(FLOATING_POINT, "score") { description = "Adamic-Adar score matrix [n, n]" }
        Composition("""
            SDVariable deg = sd.sum(adj, false, 1);
            SDVariable logDeg = sd.math().log(deg);
            SDVariable scaled = adj.div(sd.reshape(logDeg, -1, 1));
            SDVariable out = sd.mmul(adj, scaled);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Adamic-Adar link-prediction score: S[i,j] = sum_v A[i,v]·A[v,j] / log(deg_v), weighting each
            shared neighbor v by the inverse log of its degree so that rare (low-degree) common neighbors
            contribute more than hubs. Assumes node degrees > 1 (so log(deg) > 0).
            """.trimIndent()
        }
    }

    Op("resourceAllocation") {
        Input(FLOATING_POINT, "adj")    { description = "Adjacency matrix [n, n] with positive node degrees" }
        Output(FLOATING_POINT, "score") { description = "Resource-allocation score matrix [n, n]" }
        Composition("""
            SDVariable deg = sd.sum(adj, false, 1);
            SDVariable scaled = adj.div(sd.reshape(deg, -1, 1));
            SDVariable out = sd.mmul(adj, scaled);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Resource-Allocation link-prediction score: S[i,j] = sum_v A[i,v]·A[v,j] / deg_v. Like
            Adamic-Adar but penalizes high-degree shared neighbors even more strongly (inverse degree
            rather than inverse log-degree). Often the strongest of the simple topological predictors.
            """.trimIndent()
        }
    }

    Op("preferentialAttachment") {
        Input(FLOATING_POINT, "adj")    { description = "Adjacency matrix [n, n]" }
        Output(FLOATING_POINT, "score") { description = "Preferential-attachment score matrix [n, n]" }
        Composition("""
            SDVariable deg = sd.sum(adj, false, 1);
            SDVariable out = sd.mmul(sd.reshape(deg, -1, 1), sd.reshape(deg, 1, -1));
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Preferential-Attachment link-prediction score: S[i,j] = deg_i · deg_j (the outer product of
            the degree vector). Encodes the "rich get richer" hypothesis that high-degree nodes are more
            likely to acquire new links, independent of any shared neighborhood.
            """.trimIndent()
        }
    }

    Op("jaccardTopology") {
        Input(FLOATING_POINT, "adj")    { description = "Adjacency matrix [n, n] (symmetric for undirected graphs)" }
        Output(FLOATING_POINT, "score") { description = "Topological Jaccard score matrix [n, n]" }
        Composition("""
            SDVariable cn = sd.mmul(adj, adj);
            SDVariable deg = sd.sum(adj, false, 1);
            SDVariable ones = sd.onesLike(deg);
            SDVariable degI = sd.mmul(sd.reshape(deg, -1, 1), sd.reshape(ones, 1, -1));
            SDVariable degJ = sd.mmul(sd.reshape(ones, -1, 1), sd.reshape(deg, 1, -1));
            SDVariable union = degI.add(degJ).sub(cn);
            SDVariable out = cn.div(union.add(1e-9));
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Topological Jaccard link-prediction score: S[i,j] = |N(i) ∩ N(j)| / |N(i) ∪ N(j)|, computed
            as commonNeighbors(i,j) / (deg_i + deg_j - commonNeighbors(i,j)). Unlike feature-vector
            Jaccard distance, this measures overlap of graph neighborhoods, normalizing for node degree.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // Diffusion / propagation (transductive classification + link prediction)
    // -------------------------------------------------------------------------

    Op("correctAndSmooth") {
        Input(FLOATING_POINT, "basePreds") { description = "Base per-node predictions / logits [n, c]" }
        Input(FLOATING_POINT, "aNorm")     { description = "Dense (row- or sym-) normalized adjacency [n, n]" }
        Input(FLOATING_POINT, "residuals") { description = "Label residuals at training nodes, zeros elsewhere [n, c]" }
        Arg(FLOATING_POINT, "alpha1")      { description = "Residual-spreading weight in (0,1) for the Correct phase" }
        Arg(FLOATING_POINT, "alpha2")      { description = "Label-smoothing weight in (0,1) for the Smooth phase" }
        Arg(INT, "iter1")                  { description = "Number of Correct-phase propagation steps" }
        Arg(INT, "iter2")                  { description = "Number of Smooth-phase propagation steps" }
        Output(FLOATING_POINT, "out")      { description = "Corrected and smoothed per-node predictions [n, c]" }
        Composition("""
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
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Correct & Smooth post-processing (Huang et al. 2020): two-phase label diffusion over a graph.
            Phase 1 (Correct, iter1 steps): spreads label residuals E over the graph while retaining a
            fraction (1-alpha1) of the original residuals at every step, then adds the spread residuals to
            the base predictions. Phase 2 (Smooth, iter2 steps): diffuses the corrected predictions while
            retaining fraction (1-alpha2) of the corrected values. Both phases use the same normalized
            adjacency A_norm.
            """.trimIndent()
        }
    }

    Op("katzIndex") {
        Input(FLOATING_POINT, "adj")  { description = "Adjacency matrix [n, n]" }
        Arg(FLOATING_POINT, "beta")   { description = "Attenuation factor (0 < beta < 1 / spectral_radius(A))" }
        Arg(INT, "L")                 { description = "Truncation depth: number of path-length terms (>= 1)" }
        Output(FLOATING_POINT, "out") { description = "Katz similarity matrix [n, n]" }
        Composition("""
            SDVariable S = adj.mul(beta);
            SDVariable Apow = adj;
            double betaPow = beta;
            for (int l = 2; l <= L; l++) {
                Apow = sd.mmul(Apow, adj);
                betaPow = betaPow * beta;
                S = S.add(Apow.mul(betaPow));
            }
            SDVariable out = S;
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Katz similarity index (Katz 1953): a link-prediction / node-similarity score that counts all
            paths between pairs of nodes, exponentially down-weighted by path length.
            S = sum_{l=1}^{L} beta^l A^l  (truncated finite-sum approximation).
            Unlike the closed-form (I - beta*A)^{-1} - I this formulation uses only matrix multiplication
            and is fully differentiable via standard mmul backward. Requires 0 < beta < 1/spectral_radius(A)
            for the series to be meaningful; in practice beta = 0.05..0.1 and L = 3..5 works well.
            """.trimIndent()
        }
    }

    Op("simRank") {
        Input(FLOATING_POINT, "W")        { description = "Column-normalized adjacency [n, n] (each column sums to 1)" }
        Input(FLOATING_POINT, "identity") { description = "Identity matrix [n, n] -- pass sd.constant(Nd4j.eye(n))" }
        Arg(FLOATING_POINT, "C")          { description = "SimRank decay constant in (0, 1)" }
        Arg(INT, "iterations")            { description = "Number of power iterations" }
        Output(FLOATING_POINT, "out")     { description = "Node-similarity matrix [n, n]; diagonal = 1" }
        Composition("""
            SDVariable onesM = sd.onesLike(W);
            SDVariable offDiagMask = onesM.sub(identity);
            SDVariable S = identity;
            for (int i = 0; i < iterations; i++) {
                SDVariable prop = sd.mmul(sd.mmul(W, S, true, false, false), W).mul(C);
                S = prop.mul(offDiagMask).add(identity);
            }
            SDVariable out = S;
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            SimRank node-similarity (Jeh & Widom 2002): two nodes are similar if their in-neighbors are
            similar. Converges via the fixed-point iteration S_{t+1} = C * W^T S_t W (diagonal forced to 1),
            where W is the column-normalized adjacency. The diagonal reset is implemented in a gradient-clean
            elementwise form: S_new = prop * (ones - I) + I.
            Pass identity = sd.constant(Nd4j.eye(n).castTo(DataType.DOUBLE)) so its gradient is not tracked.
            """.trimIndent()
        }
    }

    Op("personalizedPageRank") {
        Input(FLOATING_POINT, "aNorm") { description = "Row-normalized adjacency [n, n]" }
        Input(FLOATING_POINT, "seed")  { description = "Personalization / seed distribution [n, c]" }
        Arg(FLOATING_POINT, "alpha")   { description = "Propagation weight (teleport = 1-alpha) in (0,1)" }
        Arg(INT, "iterations")         { description = "Number of power iterations" }
        Output(FLOATING_POINT, "out")  { description = "Personalized PageRank score matrix [n, c]" }
        Composition("""
            SDVariable r = seed;
            for (int i = 0; i < iterations; i++) {
                r = sd.mmul(aNorm, r).mul(alpha).add(seed.mul(1.0 - alpha));
            }
            SDVariable out = r;
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Personalized PageRank (PPR) via power iteration: r_{t+1} = alpha * A_norm * r_t + (1-alpha) * seed.
            At each step the walker follows graph edges (alpha) or teleports back to the personalization seed
            (1-alpha). After `iterations` steps the result approximates (I - alpha*A_norm)^{-1}*(1-alpha)*seed
            without ever forming a matrix inverse, making it fully differentiable through mmul and scalar ops.
            Use aNorm = row-normalized adjacency (each row sums to 1). For node-classification, seed = one-hot
            class indicators [n,c]; for link prediction, seed = one-hot per-query-node vectors [n,n].
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // Pooling / self-supervision / centrality
    // -------------------------------------------------------------------------

    Op("topKPool") {
        Input(FLOATING_POINT, "scores")   { description = "Per-node selection scores [n] (e.g. a learned projection)" }
        Input(FLOATING_POINT, "features") { description = "Node features [n, d]" }
        Arg(INT, "k")                     { description = "Number of nodes to keep" }
        Arg(INT, "n")                     { description = "Number of nodes (the one-hot depth)" }
        Output(FLOATING_POINT, "pooled")  { description = "Pooled features of the top-k nodes [k, d], gated by sigmoid(score)" }
        Composition("""
            SDVariable[] tk = sd.nn().topK(scores, k, false);
            SDVariable sel = sd.oneHot(tk[1], n, -1, 1.0, 0.0, features.dataType());
            SDVariable gathered = sd.mmul(sel, features);
            SDVariable gate = sd.nn().sigmoid(tk[0]);
            SDVariable out = gathered.mul(sd.expandDims(gate, -1));
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Top-k node pooling (Gao & Ji 2019 / Cangea et al. 2018; the selection mechanism of SAGPool):
            keeps the k highest-scoring nodes and gates their features by sigmoid(score) so the score stays
            differentiable. The top-k rows are gathered as oneHot(topIndices, n) @ features (avoiding a
            separate gather op). Pass scores from a learned projection or a graph-attention layer for SAGPool.
            """.trimIndent()
        }
    }


    Op("sortPool") {
        Input(FLOATING_POINT, "features") { description = "Node feature matrix [n, d]" }
        Input(FLOATING_POINT, "sortKey")  { description = "Per-node sort scores [n] (e.g. last GNN channel)" }
        Arg(INT, "k")                     { description = "Number of nodes to keep (output rows)" }
        Arg(INT, "n")                     { description = "Number of nodes (one-hot depth)" }
        Output(FLOATING_POINT, "pooled")  { description = "Top-k node features in descending sort-key order [k, d]" }
        Composition("""
            SDVariable[] tk = sd.nn().topK(sortKey, k, true);
            SDVariable sel = sd.oneHot(tk[1], n, -1, 1.0, 0.0, features.dataType());
            SDVariable out = sd.mmul(sel, features);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            SortPool (Zhang et al. 2018, DGCNN): sorts all n nodes by a scalar sort key (e.g. the last
            GNN channel) in descending order and keeps the top-k rows, producing a fixed-size graph-level
            representation [k, d] regardless of graph size. Unlike topKPool there is no sigmoid gate --
            the selection is purely order-based, which preserves the structural ordering signal used by
            DGCNN's subsequent 1-D convolution. Differentiable w.r.t. features through the top-k gather
            (oneHot @ features); the sort key selects WHICH rows are kept but its gradient is zero (as for
            topKPool / knnGraph -- it enters only through the index, not the value).
            """.trimIndent()
        }
    }

    Op("graceLoss") {
        Input(FLOATING_POINT, "z1")       { description = "Node embeddings of augmented view 1 [n, d]" }
        Input(FLOATING_POINT, "z2")       { description = "Node embeddings of augmented view 2 [n, d]" }
        Input(FLOATING_POINT, "identity") { description = "Identity matrix [n, n] -- pass sd.constant(Nd4j.eye(n))" }
        Arg(FLOATING_POINT, "tau")        { description = "Temperature (e.g. 0.5)" }
        Output(FLOATING_POINT, "loss")    { description = "Scalar GRACE / InfoNCE contrastive loss" }
        Composition("""
            SDVariable z1n = z1.div(sd.math().sqrt(sd.sum(z1.mul(z1), true, 1).add(1e-12)));
            SDVariable z2n = z2.div(sd.math().sqrt(sd.sum(z2.mul(z2), true, 1).add(1e-12)));
            SDVariable sim = sd.mmul(z1n, z2n, false, true, false).div(tau);
            SDVariable pos = sd.sum(sim.mul(identity), false, 1);
            SDVariable lse = sd.math().log(sd.sum(sd.math().exp(sim), false, 1).add(1e-12));
            SDVariable out = sd.mean(lse.sub(pos));
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            GRACE graph-contrastive loss (Zhu et al. 2020): an InfoNCE objective over two augmented views.
            For each node i, the positive pair is (z1_i, z2_i) and the negatives are all z2_j (j != i):
            loss = -mean( cosine(z1_i,z2_i)/tau - logsumexp_j cosine(z1_i,z2_j)/tau ).
            Yields label-free node embeddings. Pass identity = sd.constant(Nd4j.eye(n).castTo(DataType.DOUBLE))
            so the diagonal (positive) similarities are extracted with a gradient-clean elementwise mask.
            """.trimIndent()
        }
    }

    Op("clusteringCoefficient") {
        Input(FLOATING_POINT, "adj")      { description = "Adjacency matrix [n, n] (symmetric for undirected graphs)" }
        Input(FLOATING_POINT, "identity") { description = "Identity matrix [n, n] -- pass sd.constant(Nd4j.eye(n))" }
        Output(FLOATING_POINT, "coeff")   { description = "Per-node local clustering coefficient [n]" }
        Composition("""
            SDVariable a2 = sd.mmul(adj, adj);
            SDVariable a3 = sd.mmul(a2, adj);
            SDVariable tri = sd.sum(a3.mul(identity), false, 1);
            SDVariable deg = sd.sum(adj, false, 1);
            SDVariable denom = deg.mul(deg.sub(1.0)).add(1e-9);
            SDVariable out = tri.div(denom);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Local clustering coefficient per node: C_i = (closed triangles through i) / (possible pairs) =
            (A^3)_ii / (deg_i (deg_i - 1)). (A^3)_ii counts the length-3 closed walks through node i (= twice
            its triangle count), computed as diag(A·A·A) via an identity-mask extraction. Measures how
            tightly each node's neighborhood is interconnected. Pass identity = sd.constant(Nd4j.eye(n)).
            """.trimIndent()
        }
    }
}
