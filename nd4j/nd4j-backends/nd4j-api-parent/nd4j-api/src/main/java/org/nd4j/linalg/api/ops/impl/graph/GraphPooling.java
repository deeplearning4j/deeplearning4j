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

package org.nd4j.linalg.api.ops.impl.graph;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMax;
import org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMean;
import org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentSum;

/**
 * Graph-level (readout) pooling operations built as pure SameDiff compositions over
 * existing differentiable UnsortedSegment* ops.
 *
 * <p>Each static method aggregates per-node features into per-graph features using a
 * segment-reduce over the {@code graphIds} index vector that maps each node to its
 * containing graph.
 *
 * <p>No new native ops are introduced; gradients flow automatically through the
 * underlying {@link UnsortedSegmentSum}, {@link UnsortedSegmentMean}, and
 * {@link UnsortedSegmentMax} {@code doDiff} implementations.
 *
 * <p>Verified ctor signatures used here (all take {@code (SameDiff, SDVariable data,
 * SDVariable segmentIds, int numSegments)}):
 * <ul>
 *   <li>{@link UnsortedSegmentSum#UnsortedSegmentSum(SameDiff, SDVariable, SDVariable, int)}</li>
 *   <li>{@link UnsortedSegmentMean#UnsortedSegmentMean(SameDiff, SDVariable, SDVariable, int)}</li>
 *   <li>{@link UnsortedSegmentMax#UnsortedSegmentMax(SameDiff, SDVariable, SDVariable, int)}</li>
 * </ul>
 */
public final class GraphPooling {

    private GraphPooling() {
        // static utility class — not instantiable
    }

    /**
     * Global sum pooling: for each graph {@code g}, sums all node features whose
     * {@code graphIds} entry equals {@code g}.
     *
     * <pre>  out[g, f] = Σ_{i: graphIds[i]==g} nodeFeatures[i, f]</pre>
     *
     * @param sd           the SameDiff graph
     * @param name         output variable name; may be {@code null}
     * @param nodeFeatures per-node feature matrix [totalNodes, F]
     * @param graphIds     per-node graph assignment [totalNodes] (INT32, values in [0, numGraphs))
     * @param numGraphs    total number of graphs in the batch
     * @return per-graph feature matrix [numGraphs, F]
     */
    public static SDVariable globalSumPool(SameDiff sd, String name,
                                           SDVariable nodeFeatures, SDVariable graphIds,
                                           int numGraphs) {
        SDVariable out = sd.unsortedSegmentSum(nodeFeatures, graphIds, numGraphs);
        return (name != null) ? sd.updateVariableNameAndReference(out, name) : out;
    }

    /**
     * Global mean pooling: for each graph {@code g}, averages all node features whose
     * {@code graphIds} entry equals {@code g}.
     *
     * <pre>  out[g, f] = mean_{i: graphIds[i]==g} nodeFeatures[i, f]</pre>
     *
     * @param sd           the SameDiff graph
     * @param name         output variable name; may be {@code null}
     * @param nodeFeatures per-node feature matrix [totalNodes, F]
     * @param graphIds     per-node graph assignment [totalNodes] INT32
     * @param numGraphs    total number of graphs in the batch
     * @return per-graph feature matrix [numGraphs, F]
     */
    public static SDVariable globalMeanPool(SameDiff sd, String name,
                                            SDVariable nodeFeatures, SDVariable graphIds,
                                            int numGraphs) {
        SDVariable out = sd.unsortedSegmentMean(nodeFeatures, graphIds, numGraphs);
        return (name != null) ? sd.updateVariableNameAndReference(out, name) : out;
    }

    /**
     * Global max pooling: for each graph {@code g}, takes the element-wise maximum over
     * all node features whose {@code graphIds} entry equals {@code g}.
     *
     * <pre>  out[g, f] = max_{i: graphIds[i]==g} nodeFeatures[i, f]</pre>
     *
     * @param sd           the SameDiff graph
     * @param name         output variable name; may be {@code null}
     * @param nodeFeatures per-node feature matrix [totalNodes, F]
     * @param graphIds     per-node graph assignment [totalNodes] INT32
     * @param numGraphs    total number of graphs in the batch
     * @return per-graph feature matrix [numGraphs, F]
     */
    public static SDVariable globalMaxPool(SameDiff sd, String name,
                                           SDVariable nodeFeatures, SDVariable graphIds,
                                           int numGraphs) {
        SDVariable out = sd.unsortedSegmentMax(nodeFeatures, graphIds, numGraphs);
        return (name != null) ? sd.updateVariableNameAndReference(out, name) : out;
    }

    /**
     * Global gated-attention pooling (Li et al. 2016): each node is weighted by a learned
     * sigmoid gate and the per-graph readout is the gated sum of node features.
     *
     * <pre>
     *   gate    = sigmoid(nodeFeatures · gateWeight)          [N, 1]
     *   out[g]  = Σ_{i: graphIds[i]==g} gate[i] * nodeFeatures[i]
     * </pre>
     *
     * @param sd           the SameDiff graph
     * @param name         output variable name; may be {@code null}
     * @param nodeFeatures per-node feature matrix [totalNodes, F]
     * @param gateWeight   gate projection [F, 1]
     * @param graphIds     per-node graph assignment [totalNodes] INT32
     * @param numGraphs    total number of graphs in the batch
     * @return per-graph feature matrix [numGraphs, F]
     */
    public static SDVariable globalAttentionPool(SameDiff sd, String name,
                                                 SDVariable nodeFeatures, SDVariable gateWeight,
                                                 SDVariable graphIds, int numGraphs) {
        SDVariable gate     = sd.nn().sigmoid(sd.mmul(nodeFeatures, gateWeight));   // [N, 1]
        SDVariable weighted = nodeFeatures.mul(gate);                              // [N, F]
        SDVariable out = sd.unsortedSegmentSum(weighted, graphIds, numGraphs);
        return (name != null) ? sd.updateVariableNameAndReference(out, name) : out;
    }

    /**
     * Global softmax-attention pooling: node features are weighted by a per-graph softmax over
     * learned attention scores, then summed per graph.
     *
     * <pre>
     *   logits = nodeFeatures · scoreWeight                   [N, 1]
     *   alpha  = softmax_over_graph(logits)  (per graphIds)   [N, 1]
     *   out[g] = Σ_{i: graphIds[i]==g} alpha[i] * nodeFeatures[i]
     * </pre>
     *
     * <p>No {@code UnsortedSegmentSoftmax} op exists, so the per-graph softmax is composed from
     * segment-max / segment-sum with the standard max-subtraction for numerical stability.
     *
     * @param sd           the SameDiff graph
     * @param name         output variable name; may be {@code null}
     * @param nodeFeatures per-node feature matrix [totalNodes, F]
     * @param scoreWeight  attention-score projection [F, 1]
     * @param graphIds     per-node graph assignment [totalNodes] INT32
     * @param numGraphs    total number of graphs in the batch
     * @return per-graph feature matrix [numGraphs, F]
     */
    public static SDVariable globalSoftmaxAttentionPool(SameDiff sd, String name,
                                                        SDVariable nodeFeatures, SDVariable scoreWeight,
                                                        SDVariable graphIds, int numGraphs) {
        SDVariable logits      = sd.mmul(nodeFeatures, scoreWeight);                                       // [N, 1]
        SDVariable maxPerGraph = sd.unsortedSegmentMax(logits, graphIds, numGraphs); // [G, 1]
        SDVariable shifted     = logits.sub(sd.gather(maxPerGraph, graphIds, 0));                          // [N, 1]
        SDVariable expv        = sd.math().exp(shifted);                                                   // [N, 1]
        SDVariable sumExp      = sd.unsortedSegmentSum(expv, graphIds, numGraphs);   // [G, 1]
        SDVariable alpha       = expv.div(sd.gather(sumExp, graphIds, 0));                                 // [N, 1]
        SDVariable weighted    = nodeFeatures.mul(alpha);                                                  // [N, F]
        SDVariable out = sd.unsortedSegmentSum(weighted, graphIds, numGraphs);
        return (name != null) ? sd.updateVariableNameAndReference(out, name) : out;
    }
}
