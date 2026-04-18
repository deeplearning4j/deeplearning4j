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

package org.eclipse.deeplearning4j.llm.generation;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Tree-structured verification for speculative decoding via attention mask manipulation.
 *
 * <p>Instead of verifying a single linear chain of speculative tokens, tree attention
 * verifies multiple candidate branches in a single forward pass by constructing a
 * tree-shaped attention mask. This allows the target model to evaluate multiple
 * speculation hypotheses simultaneously, increasing the probability of accepting at
 * least one long branch.</p>
 *
 * <h3>How it works</h3>
 * <p>Given a token tree where each node can have multiple children (candidate next tokens),
 * we flatten the tree into a sequence and build a custom attention mask that encodes the
 * tree structure:</p>
 * <pre>
 *            root
 *           /    \
 *         A        B
 *        / \        |
 *       C   D       E
 *
 * Flattened: [root, A, B, C, D, E]
 * Attention mask (1=attend, 0=masked):
 *   root: [1, 0, 0, 0, 0, 0]  -- root attends to itself
 *   A:    [1, 1, 0, 0, 0, 0]  -- A attends to root and A
 *   B:    [1, 0, 1, 0, 0, 0]  -- B attends to root and B
 *   C:    [1, 1, 0, 1, 0, 0]  -- C attends to root, A, and C
 *   D:    [1, 1, 0, 0, 1, 0]  -- D attends to root, A, and D
 *   E:    [1, 0, 1, 0, 0, 1]  -- E attends to root, B, and E
 * </pre>
 *
 * <p>After the forward pass, we verify each path from root to leaf and select the
 * longest accepted path. This is more efficient than running separate forward passes
 * for each branch.</p>
 *
 * <h3>Usage with speculative decoding</h3>
 * <ol>
 *   <li>Draft model generates a tree of candidate tokens (e.g., top-2 at each position)</li>
 *   <li>TreeAttentionVerifier flattens the tree and builds the attention mask</li>
 *   <li>Target model runs one forward pass with the tree mask</li>
 *   <li>Verifier checks all paths and returns the longest accepted path</li>
 * </ol>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see Speculator
 * @see DraftModelSpeculator
 */
@Slf4j
public class TreeAttentionVerifier {

    /**
     * A node in the speculative token tree.
     *
     * <p>Each node holds a candidate token ID and references to its children
     * (alternative next-token predictions). The root node typically holds the
     * last verified token from the previous generation step.</p>
     */
    public static class TreeNode {
        @Getter
        private final int tokenId;

        @Getter
        private final int flatIndex;

        @Getter
        private final TreeNode parent;

        @Getter
        private final List<TreeNode> children;

        @Getter
        private final int depth;

        public TreeNode(int tokenId, int flatIndex, TreeNode parent, int depth) {
            this.tokenId = tokenId;
            this.flatIndex = flatIndex;
            this.parent = parent;
            this.children = new ArrayList<>();
            this.depth = depth;
        }

        /**
         * Add a child node to this node.
         *
         * @param childTokenId the candidate token ID for the child
         * @param childFlatIndex the position in the flattened sequence
         * @return the newly created child node
         */
        public TreeNode addChild(int childTokenId, int childFlatIndex) {
            TreeNode child = new TreeNode(childTokenId, childFlatIndex, this, this.depth + 1);
            children.add(child);
            return child;
        }

        /** Whether this node is a leaf (has no children). */
        public boolean isLeaf() {
            return children.isEmpty();
        }
    }

    /**
     * Result of tree-structured verification.
     */
    public static class TreeVerificationResult {
        /** The longest accepted path of token IDs (excluding root). */
        @Getter
        private final int[] acceptedTokens;

        /** The correction token at the first rejection point. */
        @Getter
        private final int correctionToken;

        /** Total number of nodes in the tree that were evaluated. */
        @Getter
        private final int totalNodesEvaluated;

        /** Number of distinct root-to-leaf paths in the tree. */
        @Getter
        private final int totalPaths;

        public TreeVerificationResult(int[] acceptedTokens, int correctionToken,
                                      int totalNodesEvaluated, int totalPaths) {
            this.acceptedTokens = acceptedTokens;
            this.correctionToken = correctionToken;
            this.totalNodesEvaluated = totalNodesEvaluated;
            this.totalPaths = totalPaths;
        }
    }

    /**
     * Build a tree attention mask for a flattened token tree.
     *
     * <p>The mask encodes tree structure so that each node attends only to its ancestors
     * and itself. Combined with the existing past KV cache attention, this allows all
     * branches to be evaluated in one forward pass.</p>
     *
     * @param root the root node of the token tree
     * @param treeSize total number of nodes in the tree (including root)
     * @param pastSeqLen length of existing KV cache (past tokens to attend to)
     * @return attention mask of shape {@code [1, 1, treeSize, pastSeqLen + treeSize]},
     *         FLOAT dtype, with {@link ModelIOConfig#MASK_FILL} for masked positions
     */
    public static INDArray buildTreeAttentionMask(TreeNode root, int treeSize, long pastSeqLen) {
        long totalLen = pastSeqLen + treeSize;
        float[] maskData = new float[(int) (treeSize * totalLen)];

        // Initialize all tree positions as masked
        Arrays.fill(maskData, ModelIOConfig.MASK_FILL);

        // For each node, unmask its ancestors and itself (in the tree portion)
        // and all past KV positions
        List<TreeNode> flatNodes = new ArrayList<>(treeSize);
        flattenTree(root, flatNodes);

        for (TreeNode node : flatNodes) {
            int queryIdx = node.getFlatIndex();
            int rowOffset = (int) (queryIdx * totalLen);

            // Attend to all past KV cache positions
            for (int k = 0; k < pastSeqLen; k++) {
                maskData[rowOffset + k] = 0.0f;
            }

            // Attend to self
            maskData[rowOffset + (int) pastSeqLen + queryIdx] = 0.0f;

            // Attend to all ancestors
            TreeNode ancestor = node.getParent();
            while (ancestor != null) {
                maskData[rowOffset + (int) pastSeqLen + ancestor.getFlatIndex()] = 0.0f;
                ancestor = ancestor.getParent();
            }
        }

        return Nd4j.createFromArray(maskData).reshape(1, 1, treeSize, totalLen);
    }

    /**
     * Build a position ID tensor for tree-structured positions.
     *
     * <p>Each node in the tree has a position equal to {@code pastSeqLen + depth},
     * since nodes at the same tree depth are at the same logical sequence position
     * (they represent alternative tokens for the same step).</p>
     *
     * @param root the root node of the token tree
     * @param treeSize total number of nodes
     * @param pastSeqLen length of existing KV cache
     * @return position IDs of shape {@code [1, treeSize]}, LONG dtype
     */
    public static INDArray buildTreePositionIds(TreeNode root, int treeSize, long pastSeqLen) {
        long[] posIds = new long[treeSize];
        List<TreeNode> flatNodes = new ArrayList<>(treeSize);
        flattenTree(root, flatNodes);

        for (TreeNode node : flatNodes) {
            posIds[node.getFlatIndex()] = pastSeqLen + node.getDepth();
        }

        return Nd4j.createFromArray(posIds).reshape(1, treeSize);
    }

    /**
     * Flatten the flattened token sequence from the tree (for embedding lookup).
     *
     * @param root the root node
     * @param treeSize total number of nodes
     * @return array of token IDs in flattened order
     */
    public static int[] flattenTokenIds(TreeNode root, int treeSize) {
        int[] tokenIds = new int[treeSize];
        List<TreeNode> flatNodes = new ArrayList<>(treeSize);
        flattenTree(root, flatNodes);

        for (TreeNode node : flatNodes) {
            tokenIds[node.getFlatIndex()] = node.getTokenId();
        }

        return tokenIds;
    }

    /**
     * Verify all paths in the token tree against the target model's logits.
     *
     * <p>For each root-to-leaf path, checks how many tokens the target model agrees
     * with via greedy decoding. Returns the longest accepted path.</p>
     *
     * @param root the root node of the token tree
     * @param treeSize total number of nodes
     * @param logits target model logits, shape {@code [1, treeSize, vocabSize]}
     * @return the verification result with the longest accepted path
     */
    public static TreeVerificationResult verifyTree(TreeNode root, int treeSize, INDArray logits) {
        // Collect all root-to-leaf paths
        List<List<TreeNode>> paths = new ArrayList<>();
        collectPaths(root, new ArrayList<>(), paths);

        int bestAccepted = -1;
        int[] bestTokens = new int[0];
        int bestCorrectionToken = -1;

        for (List<TreeNode> path : paths) {
            // path[0] is the root; path[1..] are the speculative tokens to verify
            int accepted = 0;
            int correctionToken = -1;

            for (int i = 0; i < path.size() - 1; i++) {
                TreeNode current = path.get(i);
                TreeNode next = path.get(i + 1);

                // The logits at current's position predict the next token
                int modelToken = argmaxAtPosition(logits, current.getFlatIndex());

                if (modelToken == next.getTokenId()) {
                    accepted++;
                } else {
                    correctionToken = modelToken;
                    break;
                }
            }

            // If all tokens in path accepted, correction comes from the last node's logits
            if (correctionToken < 0 && path.size() > 1) {
                TreeNode lastNode = path.get(path.size() - 1);
                correctionToken = argmaxAtPosition(logits, lastNode.getFlatIndex());
            } else if (correctionToken < 0) {
                correctionToken = argmaxAtPosition(logits, root.getFlatIndex());
            }

            if (accepted > bestAccepted) {
                bestAccepted = accepted;
                bestCorrectionToken = correctionToken;

                // Build accepted token array (path[1] through path[accepted], excluding root)
                bestTokens = new int[accepted];
                for (int i = 0; i < accepted; i++) {
                    bestTokens[i] = path.get(i + 1).getTokenId();
                }
            }
        }

        if (bestAccepted < 0) {
            bestAccepted = 0;
            bestCorrectionToken = argmaxAtPosition(logits, root.getFlatIndex());
        }

        return new TreeVerificationResult(bestTokens, bestCorrectionToken, treeSize, paths.size());
    }

    /**
     * Build a simple token tree from a draft model's top-k predictions at each depth.
     *
     * <p>Creates a balanced tree where each node has {@code branchFactor} children,
     * up to {@code maxDepth} levels deep. The token IDs at each position come from
     * the provided predictions array.</p>
     *
     * @param rootTokenId the root token ID (last verified token)
     * @param predictions predicted tokens, shape {@code [maxDepth][branchFactor]}
     * @param maxDepth maximum tree depth
     * @param branchFactor number of children per node
     * @return a pair of (root node, total tree size)
     */
    public static TreeBuildResult buildBalancedTree(
            int rootTokenId, int[][] predictions, int maxDepth, int branchFactor) {

        int nextFlatIndex = 0;
        TreeNode root = new TreeNode(rootTokenId, nextFlatIndex++, null, 0);

        List<TreeNode> currentLevel = new ArrayList<>();
        currentLevel.add(root);

        for (int depth = 0; depth < maxDepth && depth < predictions.length; depth++) {
            List<TreeNode> nextLevel = new ArrayList<>();
            int numBranches = Math.min(branchFactor, predictions[depth].length);

            for (TreeNode parent : currentLevel) {
                for (int b = 0; b < numBranches; b++) {
                    TreeNode child = parent.addChild(predictions[depth][b], nextFlatIndex++);
                    nextLevel.add(child);
                }
            }
            currentLevel = nextLevel;
        }

        return new TreeBuildResult(root, nextFlatIndex);
    }

    /**
     * Result of building a token tree.
     */
    public static class TreeBuildResult {
        @Getter
        private final TreeNode root;

        @Getter
        private final int treeSize;

        public TreeBuildResult(TreeNode root, int treeSize) {
            this.root = root;
            this.treeSize = treeSize;
        }
    }

    // ========== Internal Helpers ==========

    /**
     * Flatten the tree into a list in flat-index order.
     */
    private static void flattenTree(TreeNode node, List<TreeNode> result) {
        // Ensure the list is large enough
        while (result.size() <= node.getFlatIndex()) {
            result.add(null);
        }
        result.set(node.getFlatIndex(), node);

        for (TreeNode child : node.getChildren()) {
            flattenTree(child, result);
        }
    }

    /**
     * Collect all root-to-leaf paths in the tree.
     */
    private static void collectPaths(TreeNode node, List<TreeNode> currentPath, List<List<TreeNode>> allPaths) {
        currentPath.add(node);

        if (node.isLeaf()) {
            allPaths.add(new ArrayList<>(currentPath));
        } else {
            for (TreeNode child : node.getChildren()) {
                collectPaths(child, currentPath, allPaths);
            }
        }

        currentPath.remove(currentPath.size() - 1);
    }

    /**
     * Extract argmax token from logits at a specific flat position.
     */
    private static int argmaxAtPosition(INDArray logits, int position) {
        INDArray posLogits;
        if (logits.rank() == 3) {
            posLogits = logits.get(NDArrayIndex.point(0), NDArrayIndex.point(position), NDArrayIndex.all());
        } else if (logits.rank() == 2) {
            posLogits = logits.get(NDArrayIndex.point(position), NDArrayIndex.all());
        } else {
            posLogits = logits;
        }
        return SamplerUtils.argmax(posLogits);
    }
}
