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

package org.eclipse.deeplearning4j.llm;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * BRANCH PARITY oracle for optimum merged decoders (ONNX If, use_cache_branch):
 * the no-past (else) and with-past (then) branches implement THE SAME function,
 * so for an identical token prefix over identical (random, seeded) encoder
 * states, teacher-forced else-branch logits and stepwise then-branch logits
 * must agree position-for-position. The first divergent position localizes the
 * with-past branch defect (observed on whisper: incremental decode opens
 * correctly then drifts into fluent hallucination while full-recompute through
 * the else branch is word-perfect).
 *
 * <p>No audio involved: parity is a property of the graph, not the content.
 */
@Slf4j
@Tag("llm")
public class WhisperMergedBranchParityTest {

    private static final File MERGED = new File(System.getProperty("user.home"),
            ".cache/dl4j-whisper-models/whisper-tiny-onnx/decoder_model_merged.onnx");

    // [SOT, en, transcribe, notimestamps] + "The"," quick","s"," are"
    private static final long[] PREFIX = {50258, 50259, 50359, 50363, 440, 1702, 82, 366};
    private static final int PROMPT_LEN = 4;
    private static final int LAYERS = 4;

    @Test
    public void testThenElseBranchLogitParity() {
        assumeTrue(MERGED.exists(), "whisper-tiny merged decoder not cached");

        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        // Same flags as WhisperModel.fromOnnx: no suggested-shape pinning, no tracking.
        SameDiff decoder = importer.runImport(MERGED.getAbsolutePath(),
                Collections.emptyMap(), false, false);

        Nd4j.getRandom().setSeed(42);
        INDArray enc = Nd4j.rand(DataType.FLOAT, 1, 1500, 384).subi(0.5);

        List<String> outNames = new ArrayList<>();
        outNames.add("logits");
        List<String> presentNames = new ArrayList<>();
        for (int l = 0; l < LAYERS; l++) {
            for (String kind : new String[]{"decoder", "encoder"}) {
                for (String kv : new String[]{"key", "value"}) {
                    presentNames.add("present." + l + "." + kind + "." + kv);
                }
            }
        }
        outNames.addAll(presentNames);
        String[] outArr = outNames.toArray(new String[0]);

        // ── Path A: whole prefix teacher-forced through the ELSE (no-past) branch ──
        Map<String, INDArray> mapA = new LinkedHashMap<>();
        mapA.put("input_ids", Nd4j.createFromArray(new long[][]{PREFIX}));
        mapA.put("encoder_hidden_states", enc);
        mapA.put("use_cache_branch", Nd4j.scalar(false));
        Map<String, INDArray> outA = decoder.output(mapA, outArr);
        INDArray logitsA = outA.get("logits").dup();
        log.info("Path A (else, teacher-forced) logits: {}", java.util.Arrays.toString(logitsA.shape()));

        // Repeat-call sanity: the SAME inputs through the SAME graph a second time.
        // Divergence here means execution MUTATES graph/session state (constants,
        // cached values, eager arrays) — corruption independent of any branch logic.
        Map<String, INDArray> outA2 = decoder.output(new LinkedHashMap<>(mapA), outArr);
        double repeatDiff = maxAbsDiff(logitsA, outA2.get("logits"));
        log.info("REPEAT-CALL check (A vs A2): logits maxAbsDiff={} presentDec0Shape={} presentEnc0Shape={}",
                repeatDiff,
                java.util.Arrays.toString(outA2.get("present.0.decoder.key").shape()),
                java.util.Arrays.toString(outA2.get("present.0.encoder.key").shape()));

        // ── Path B: prompt prefill (else) then stepwise THEN branch ──
        long[] prompt = java.util.Arrays.copyOf(PREFIX, PROMPT_LEN);
        Map<String, INDArray> mapP = new LinkedHashMap<>();
        mapP.put("input_ids", Nd4j.createFromArray(new long[][]{prompt}));
        mapP.put("encoder_hidden_states", enc);
        mapP.put("use_cache_branch", Nd4j.scalar(false));
        Map<String, INDArray> outP = decoder.output(mapP, outArr);

        // Prefill last-position logits must equal path A at PROMPT_LEN-1.
        double prefillDiff = maxAbsDiff(
                outA.get("logits").get(point(0), point(PROMPT_LEN - 1), all()),
                outP.get("logits").get(point(0), point(PROMPT_LEN - 1), all()));
        log.info("Parity pos {} (prefill vs A): maxAbsDiff={}", PROMPT_LEN - 1, prefillDiff);

        Map<String, INDArray> pasts = new LinkedHashMap<>();
        for (String p : presentNames) {
            pasts.put(p.replace("present", "past_key_values"), outP.get(p).dup());
        }

        // Position probe: the then-frame position-embedding slice. Identical VALUES
        // across steps = the position derivation is frozen at the first step's
        // shape (execution-baked variable shapes feeding shape_of/position math).
        String posSliceVar = null;
        for (String v : decoder.variableMap().keySet()) {
            if (v.contains("/then/") && v.contains("embed_positions/Slice") && v.endsWith("_output_0")) {
                posSliceVar = v;
                break;
            }
        }
        log.info("Position probe variable: {}", posSliceVar);
        String[] outArrB = outArr;
        if (posSliceVar != null) {
            List<String> withProbe = new ArrayList<>(outNames);
            withProbe.add(posSliceVar);
            outArrB = withProbe.toArray(new String[0]);
        }
        INDArray prevPosSlice = null;

        boolean allMatch = prefillDiff < 1e-3;
        INDArray prevStepLogits = null;
        for (int k = PROMPT_LEN; k < PREFIX.length; k++) {
            Map<String, INDArray> mapB = new LinkedHashMap<>(pasts);
            mapB.put("input_ids", Nd4j.createFromArray(new long[][]{{PREFIX[k]}}));
            mapB.put("encoder_hidden_states", enc);
            mapB.put("use_cache_branch", Nd4j.scalar(true));
            Map<String, INDArray> outB = decoder.output(mapB, outArrB);
            if (posSliceVar != null && outB.get(posSliceVar) != null) {
                INDArray posSlice = outB.get(posSliceVar);
                log.info("  step {} posSlice shape={} amean={} sameAsPrev={}",
                        k, java.util.Arrays.toString(posSlice.shape()),
                        String.format("%.5f", posSlice.amean().getDouble(0)),
                        prevPosSlice == null ? "-" : String.valueOf(maxAbsDiff(prevPosSlice, posSlice) == 0.0));
                prevPosSlice = posSlice.dup();
            }

            INDArray stepLogits = outB.get("logits").get(point(0), point(0), all());
            INDArray refLogits = logitsA.get(point(0), point(k), all());
            double d = maxAbsDiff(refLogits, stepLogits);
            int argmaxRef = refLogits.argMax().getInt(0);
            int argmaxStep = stepLogits.argMax().getInt(0);
            log.info("Parity pos {} (then-step vs A): maxAbsDiff={} argmax A={} B={} {}",
                    k, String.format("%.5f", d), argmaxRef, argmaxStep,
                    d < 1e-3 ? "OK" : "*** DIVERGED ***");
            if (d >= 1e-3) {
                allMatch = false;
                // Stale-vs-miscompute: identical to the PREVIOUS step's logits means
                // the session served stale results; different means real compute error.
                if (prevStepLogits != null) {
                    log.info("  step {} logits vs PREV step logits: maxAbsDiff={}",
                            k, maxAbsDiff(prevStepLogits, stepLogits));
                }
                // Concat-prefix consistency: present[:, :, :pastLen] must equal fed past.
                INDArray fedDec0 = pasts.get("past_key_values.0.decoder.key");
                INDArray outDec0 = outB.get("present.0.decoder.key");
                if (fedDec0 != null && outDec0 != null && outDec0.shape()[2] == fedDec0.shape()[2] + 1) {
                    INDArray prefixRows = outDec0.get(all(), all(),
                            org.nd4j.linalg.indexing.NDArrayIndex.interval(0, fedDec0.shape()[2]), all());
                    log.info("  step {} dec0 concat-prefix vs fed past: maxAbsDiff={}",
                            k, maxAbsDiff(fedDec0, prefixRows));
                } else {
                    log.info("  step {} dec0 shapes: fed={} out={}", k,
                            fedDec0 == null ? "null" : java.util.Arrays.toString(fedDec0.shape()),
                            outDec0 == null ? "null" : java.util.Arrays.toString(outDec0.shape()));
                }
            }
            prevStepLogits = stepLogits.dup();

            // Ground-truth KV check: this step's FULL output presents vs path A's
            // presents sliced to the same length. Prefix rows already proven equal —
            // this catches a corrupt NEWLY-APPENDED row (poisons the NEXT step even
            // when THIS step's logits are perfect).
            for (String p : new String[]{"present.0.decoder.key", "present.1.decoder.key",
                    "present.2.decoder.key", "present.3.decoder.key", "present.3.decoder.value",
                    "present.0.encoder.key", "present.0.encoder.value"}) {
                INDArray outPres = outB.get(p);
                INDArray refPres = outA.get(p);
                if (p.contains("encoder")) {
                    // Cross-attn presents are passthrough: must equal A's EXACTLY.
                    log.info("  step {} '{}': outShape={} refShape={} diff={}",
                            k, p,
                            outPres == null ? "null" : java.util.Arrays.toString(outPres.shape()),
                            refPres == null ? "null" : java.util.Arrays.toString(refPres.shape()),
                            outPres != null && refPres != null
                                    && java.util.Arrays.equals(outPres.shape(), refPres.shape())
                                    ? String.format("%.5f", maxAbsDiff(refPres, outPres)) : "shape-mismatch");
                    continue;
                }
                if (outPres != null && refPres != null && refPres.shape()[2] >= outPres.shape()[2]) {
                    INDArray refSlice = refPres.get(all(), all(),
                            org.nd4j.linalg.indexing.NDArrayIndex.interval(0, outPres.shape()[2]), all());
                    INDArray newRowOut = outPres.get(all(), all(), point(outPres.shape()[2] - 1), all());
                    INDArray newRowRef = refPres.get(all(), all(), point(outPres.shape()[2] - 1), all());
                    log.info("  step {} '{}' vs A-slice: fullDiff={} newRowDiff={}",
                            k, p,
                            String.format("%.5f", maxAbsDiff(refSlice, outPres)),
                            String.format("%.5f", maxAbsDiff(newRowRef, newRowOut)));
                }
            }

            Map<String, INDArray> nextPasts = new LinkedHashMap<>();
            for (String p : presentNames) {
                { INDArray pres = outB.get(p); String pn = p.replace("present", "past_key_values"); nextPasts.put(pn, pres == null || pres.isEmpty() || pres.length() == 0 ? pasts.get(pn) : pres.dup()); }
            }
            pasts = nextPasts;
        }

        // ── Layer-0 interior sweep at the first divergent step ──
        // present.0 exact + present.1 wrong localizes corruption to layer 0's
        // self-attn/cross-attn/MLP. Compare every layer-0 then-frame intermediate
        // against its else-frame twin at the same position; first mismatch = the op.
        {
            int k = PROMPT_LEN + 1; // first divergent step
            List<String> thenVars = new ArrayList<>();
            for (String v : decoder.variableMap().keySet()) {
                if (v.contains("/then//model/decoder/layers.0/") && v.endsWith("_output_0")
                        && !v.contains("Constant") && decoder.hasVariable(v.replace("/then/", "/else/"))) {
                    thenVars.add(v);
                }
            }
            Collections.sort(thenVars);
            log.info("Layer-0 sweep at step {}: {} paired intermediates", k, thenVars.size());

            // Rebuild pasts to step k (ground truth from A to eliminate accumulation).
            Map<String, INDArray> gtPasts = new LinkedHashMap<>();
            for (String p : presentNames) {
                INDArray ref = outA.get(p);
                INDArray sliced = p.contains("encoder") ? ref
                        : ref.get(all(), all(), org.nd4j.linalg.indexing.NDArrayIndex.interval(0, k), all());
                // Deliberately feed the RAW interval-slice VIEW: verifies the session-boundary
                // view materialization (preprocessPlaceholders) — with it, this sweep stays ~1e-5;
                // without it, native ops mis-read the strides and every number here lies.
                gtPasts.put(p.replace("present", "past_key_values"), sliced);
            }
            Map<String, INDArray> mapS = new LinkedHashMap<>(gtPasts);
            mapS.put("input_ids", Nd4j.createFromArray(new long[][]{{PREFIX[k]}}));
            mapS.put("encoder_hidden_states", enc);
            mapS.put("use_cache_branch", Nd4j.scalar(true));
            List<String> sweepOuts = new ArrayList<>(thenVars);
            sweepOuts.add("logits");
            Map<String, INDArray> outS = decoder.output(mapS, sweepOuts.toArray(new String[0]));

            Map<String, INDArray> mapE = new LinkedHashMap<>();
            mapE.put("input_ids", Nd4j.createFromArray(new long[][]{java.util.Arrays.copyOf(PREFIX, k + 1)}));
            mapE.put("encoder_hidden_states", enc);
            mapE.put("use_cache_branch", Nd4j.scalar(false));
            List<String> elseOuts = new ArrayList<>();
            for (String tv : thenVars) elseOuts.add(tv.replace("/then/", "/else/"));
            elseOuts.add("logits");
            Map<String, INDArray> outE = decoder.output(mapE, elseOuts.toArray(new String[0]));

            log.info("GT-pasts step {} logits diff vs else: {}", k,
                    maxAbsDiff(outE.get("logits").get(point(0), point(k), all()),
                            outS.get("logits").get(point(0), point(0), all())));
            for (String tv : thenVars) {
                INDArray tArr = outS.get(tv);
                INDArray eArr = outE.get(tv.replace("/then/", "/else/"));
                if (tArr == null || eArr == null) continue;
                String verdict;
                if (java.util.Arrays.equals(tArr.shape(), eArr.shape())) {
                    verdict = String.format("%.5f", maxAbsDiff(eArr, tArr));
                } else if (tArr.rank() == eArr.rank() && tArr.rank() >= 2
                        && tArr.shape()[1] == 1 && eArr.shape()[1] == k + 1
                        && java.util.Arrays.equals(
                                eArr.get(all(), point(k)).shape(), tArr.get(all(), point(0)).shape())) {
                    verdict = String.format("%.5f (lastpos)", maxAbsDiff(
                            eArr.get(all(), point(k)), tArr.get(all(), point(0))));
                } else if (tArr.rank() == eArr.rank() && tArr.rank() >= 3
                        && tArr.shape()[2] == 1 && eArr.shape()[2] == k + 1
                        && java.util.Arrays.equals(
                                eArr.get(all(), all(), point(k)).shape(), tArr.get(all(), all(), point(0)).shape())) {
                    verdict = String.format("%.5f (lastpos d2)", maxAbsDiff(
                            eArr.get(all(), all(), point(k)), tArr.get(all(), all(), point(0))));
                } else {
                    verdict = "shape " + java.util.Arrays.toString(tArr.shape())
                            + " vs " + java.util.Arrays.toString(eArr.shape());
                }
                log.info("  L0 '{}': {}", tv.substring(tv.indexOf("layers.0/") + 9), verdict);
            }

            // Nested-If autopsy: ancestor chain of the then-frame's encoder-present
            // (the tensor that comes out EMPTY at pred=true). Walk graph structure,
            // request every ancestor in one call, print states — names the op where
            // the passthrough dies.
            {
                String target = null;
                for (String v : decoder.variableMap().keySet()) {
                    if (v.equals("optimum::if/then/present.0.encoder.key")) { target = v; break; }
                }
                if (target == null) {
                    for (String v : decoder.variableMap().keySet()) {
                        if (v.contains("/then/") && v.endsWith("present.0.encoder.key")) { target = v; break; }
                    }
                }
                log.info("Nested-If autopsy target: {}", target);
                if (target != null) {
                    org.nd4j.autodiff.samediff.internal.Variable tm = decoder.getVariables().get(target);
                    INDArray stored = tm.getVariable().getArr();
                    log.info("Target meta: type={} outputOfOp={} storedArr={} inputsForOp={}",
                            tm.getVariable().getVariableType(), tm.getOutputOfOp(),
                            stored == null ? "null" : java.util.Arrays.toString(stored.shape())
                                    + (stored.isEmpty() ? "(EMPTY)" : ""),
                            tm.getInputsForOp());
                    int innerSwitches = 0, innerMerges = 0;
                    for (Map.Entry<String, org.nd4j.autodiff.samediff.internal.SameDiffOp> e2
                            : decoder.getOps().entrySet()) {
                        if (!e2.getKey().contains("/then/")) continue;
                        Object opObj = e2.getValue().getOp();
                        if (opObj instanceof org.nd4j.linalg.api.ops.impl.controlflow.compat.Switch) {
                            if (innerSwitches++ < 4) log.info("  then-frame Switch: {}", e2.getKey());
                        } else if (opObj instanceof org.nd4j.linalg.api.ops.impl.controlflow.compat.Merge) {
                            if (innerMerges++ < 4) log.info("  then-frame Merge: {}", e2.getKey());
                        }
                    }
                    log.info("Then-frame control-flow ops: {} switches, {} merges", innerSwitches, innerMerges);
                }
                if (target != null) {
                    List<String> chain = new ArrayList<>();
                    java.util.Set<String> seen = new java.util.HashSet<>();
                    java.util.Deque<String> queue = new java.util.ArrayDeque<>();
                    queue.add(target);
                    while (!queue.isEmpty() && chain.size() < 30) {
                        String v = queue.poll();
                        if (!seen.add(v)) continue;
                        chain.add(v);
                        org.nd4j.autodiff.samediff.internal.Variable m = decoder.getVariables().get(v);
                        String prod = m != null ? m.getOutputOfOp() : null;
                        if (prod == null) continue;
                        List<String> ins = decoder.getOps().get(prod).getInputsToOp();
                        if (ins != null) queue.addAll(ins);
                    }
                    Map<String, INDArray> outC = decoder.output(new LinkedHashMap<>(mapS),
                            chain.toArray(new String[0]));
                    for (String v : chain) {
                        INDArray arr = outC.get(v);
                        org.nd4j.autodiff.samediff.internal.Variable m = decoder.getVariables().get(v);
                        String prod = m != null ? m.getOutputOfOp() : null;
                        String opType = prod != null && decoder.getOps().get(prod) != null
                                && decoder.getOps().get(prod).getOp() != null
                                ? decoder.getOps().get(prod).getOp().opName() : "-";
                        log.info("  CHAIN {} <- {} : {}", v.replace("optimum::if/", ""), opType,
                                arr == null ? "NULL"
                                        : (arr.isEmpty() ? "EMPTY" + java.util.Arrays.toString(arr.shape())
                                        : java.util.Arrays.toString(arr.shape())));
                    }
                }
            }

            // Direct interrogation of the smoking op: the then-frame score MatMul's
            // DECLARED inputs (graph truth, no name-pairing guesses). Dump both, and
            // match the K-side tensor against permutations of ground-truth K.
            String thenScores = null;
            for (String v : thenVars) {
                if (v.endsWith("self_attn/MatMul_output_0")) { thenScores = v; break; }
            }
            org.nd4j.autodiff.samediff.internal.Variable scoreMeta =
                    decoder.getVariables().get(thenScores);
            String scoreOp = scoreMeta.getOutputOfOp();
            List<String> scoreInputs = decoder.getOps().get(scoreOp).getInputsToOp();
            log.info("Score MatMul op '{}' inputs: {}", scoreOp, scoreInputs);

            Map<String, INDArray> mapS2 = new LinkedHashMap<>(mapS);
            List<String> probeOuts = new ArrayList<>(scoreInputs);
            probeOuts.add(thenScores);
            Map<String, INDArray> outS2 = decoder.output(mapS2, probeOuts.toArray(new String[0]));
            INDArray qIn = outS2.get(scoreInputs.get(0));
            INDArray kIn = outS2.get(scoreInputs.get(1));
            log.info("Q-side '{}': shape={}", scoreInputs.get(0),
                    qIn == null ? "null" : java.util.Arrays.toString(qIn.shape()));
            log.info("K-side '{}': shape={}", scoreInputs.get(1),
                    kIn == null ? "null" : java.util.Arrays.toString(kIn.shape()));

            // Ground-truth K at this step: A's present.0.decoder.key [1,6,k+1,64].
            INDArray gtK = outA.get("present.0.decoder.key")
                    .get(all(), all(), org.nd4j.linalg.indexing.NDArrayIndex.interval(0, k + 1), all());
            if (kIn != null) {
                long[][] perms = {{0,1,2,3},{0,1,3,2},{1,0,2,3},{1,0,3,2}};
                for (long[] pm : perms) {
                    INDArray permuted = gtK.permute((int) pm[0], (int) pm[1], (int) pm[2], (int) pm[3]);
                    if (java.util.Arrays.equals(permuted.shape(),
                            kIn.rank() == 4 ? kIn.shape() : new long[]{-1})) {
                        log.info("  K-side vs GT-K permute{}: maxAbsDiff={}",
                                java.util.Arrays.toString(pm),
                                String.format("%.5f", maxAbsDiff(permuted.dup(), kIn)));
                    } else if (kIn.rank() == 3 && permuted.shape()[0] == 1) {
                        INDArray squeezed = permuted.dup().reshape(permuted.shape()[1],
                                permuted.shape()[2], permuted.shape()[3]);
                        if (java.util.Arrays.equals(squeezed.shape(), kIn.shape())) {
                            log.info("  K-side vs GT-K permute{} (squeezed): maxAbsDiff={}",
                                    java.util.Arrays.toString(pm),
                                    String.format("%.5f", maxAbsDiff(squeezed, kIn)));
                        }
                    }
                }
                // Score row: recompute expected scores from Q-side and GT-K directly.
                if (qIn != null) {
                    INDArray tS = outS2.get(thenScores);
                    log.info("Score out shape={} head0 row={}",
                            java.util.Arrays.toString(tS.shape()),
                            java.util.Arrays.toString(java.util.Arrays.copyOf(
                                    tS.ravel().toDoubleVector(), Math.min(8, (int) tS.length()))));
                }

                // One hop up: the K-side producer (Transpose_3) and ITS producer
                // (Reshape). Dump the reshape's DATA input (should equal the concat =
                // presents = ground truth) and its TARGET-SHAPE VECTOR VALUES —
                // a 6/64 swap there is a shape-assembly bug; correct target with
                // scrambled values is the native reshape-view corruption class.
                String kProducer = decoder.getVariables().get(scoreInputs.get(1)).getOutputOfOp();
                List<String> kProdIns = decoder.getOps().get(kProducer).getInputsToOp();
                log.info("K producer '{}' inputs: {}", kProducer, kProdIns);
                String reshapeVar = kProdIns.get(0);
                String reshapeOp = decoder.getVariables().get(reshapeVar).getOutputOfOp();
                List<String> reshapeIns = reshapeOp != null ? decoder.getOps().get(reshapeOp).getInputsToOp() : null;
                log.info("Reshape op '{}' inputs: {}", reshapeOp, reshapeIns);
                if (reshapeIns != null && reshapeIns.size() >= 2) {
                    List<String> deep = new ArrayList<>(reshapeIns);
                    deep.add(reshapeVar);
                    Map<String, INDArray> outS3 = decoder.output(new LinkedHashMap<>(mapS),
                            deep.toArray(new String[0]));
                    INDArray reshapeData = outS3.get(reshapeIns.get(0));
                    INDArray reshapeShape = outS3.get(reshapeIns.get(1));
                    INDArray reshapeOut = outS3.get(reshapeVar);
                    log.info("Reshape DATA shape={} vs GT-K diff={}",
                            reshapeData == null ? "null" : java.util.Arrays.toString(reshapeData.shape()),
                            reshapeData != null && java.util.Arrays.equals(reshapeData.shape(), gtK.shape())
                                    ? String.format("%.5f", maxAbsDiff(gtK.dup(), reshapeData)) : "shape-mismatch");
                    log.info("Reshape TARGET-SHAPE values={} out shape={}",
                            reshapeShape == null ? "null" : java.util.Arrays.toString(reshapeShape.toLongVector()),
                            reshapeOut == null ? "null" : java.util.Arrays.toString(reshapeOut.shape()));
                }
            }
        }

        // ── Discriminator pass: fresh InferenceSession per step ──
        // Parity restored here = per-SESSION state bakes at the first with-past call
        // (DAG cache / cached const values / configure side effects held by the
        // session). Still diverging = graph-level bake on the op objects themselves.
        log.info("--- fresh-session-per-step pass ---");
        Map<String, INDArray> pasts2 = new LinkedHashMap<>();
        for (String p : presentNames) {
            pasts2.put(p.replace("present", "past_key_values"), outP.get(p).dup());
        }
        boolean freshAllMatch = true;
        for (int k = PROMPT_LEN; k < PREFIX.length; k++) {
            clearSessions(decoder);
            Map<String, INDArray> mapB = new LinkedHashMap<>(pasts2);
            mapB.put("input_ids", Nd4j.createFromArray(new long[][]{{PREFIX[k]}}));
            mapB.put("encoder_hidden_states", enc);
            mapB.put("use_cache_branch", Nd4j.scalar(true));
            Map<String, INDArray> outB = decoder.output(mapB, outArr);
            INDArray stepLogits = outB.get("logits").get(point(0), point(0), all());
            double d = maxAbsDiff(logitsA.get(point(0), point(k), all()), stepLogits);
            log.info("FRESH-SESSION parity pos {}: maxAbsDiff={} {}",
                    k, String.format("%.5f", d), d < 1e-3 ? "OK" : "*** DIVERGED ***");
            if (d >= 1e-3) freshAllMatch = false;
            Map<String, INDArray> nextPasts = new LinkedHashMap<>();
            for (String p : presentNames) {
                { INDArray pres = outB.get(p); String pn = p.replace("present", "past_key_values"); nextPasts.put(pn, pres == null || pres.isEmpty() || pres.length() == 0 ? pasts2.get(pn) : pres.dup()); }
            }
            pasts2 = nextPasts;
        }
        log.info("VERDICT: sharedSession={} freshSessionPerStep={}",
                allMatch ? "PARITY" : "DIVERGED", freshAllMatch ? "PARITY" : "DIVERGED");

        assertTrue(allMatch, "then-branch stepwise logits diverged from else-branch teacher forcing — see log");
    }

    private static void clearSessions(SameDiff sd) {
        try {
            java.lang.reflect.Field f = SameDiff.class.getDeclaredField("sessions");
            f.setAccessible(true);
            ((Map<?, ?>) f.get(sd)).clear();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static double maxAbsDiff(INDArray a, INDArray b) {
        return a.sub(b.castTo(a.dataType())).amax().getDouble(0);
    }

    private static org.nd4j.linalg.indexing.INDArrayIndex point(long i) {
        return org.nd4j.linalg.indexing.NDArrayIndex.point(i);
    }

    private static org.nd4j.linalg.indexing.INDArrayIndex all() {
        return org.nd4j.linalg.indexing.NDArrayIndex.all();
    }
}
