/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  ******************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Shared fixtures and lifecycle helpers for the split external-input DSP tests.
 *
 * <p>The individual classes own their behavioral scenarios; this class owns only
 * the intentionally identical toy graphs and setup mechanics. Keeping those
 * mechanics here prevents the extracted suites from drifting while preserving
 * the original test coverage.</p>
 */
abstract class DspExtInputTestSupport {

    /** Single placeholder x → matmul(w) + b → out. */
    protected final SameDiff buildSinglePlaceholder(int inDim, int outDim) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, inDim);
        SDVariable w = g.var("w", Transforms.abs(
                Nd4j.randn(DataType.FLOAT, inDim, outDim)).addi(0.1f));
        SDVariable b = g.var("b", Nd4j.ones(DataType.FLOAT, 1, outDim));
        SDVariable mm = g.mmul("mm", x, w);
        mm.add("out", b);
        return g;
    }

    /** Single placeholder graph with caller-supplied reference weights. */
    protected final SameDiff buildSinglePlaceholder(int inDim, int outDim,
                                                      INDArray wArr, INDArray bArr) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, inDim);
        SDVariable w = g.var("w", wArr.dup());
        SDVariable b = g.var("b", bArr.dup());
        SDVariable mm = g.mmul("mm", x, w);
        mm.add("out", b);
        return g;
    }

    /** Multi-placeholder matmul(x, w) + b → out. */
    protected final SameDiff buildMultiPlaceholder(int inDim, int outDim) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, inDim);
        SDVariable w = g.placeHolder("w", DataType.FLOAT, inDim, outDim);
        SDVariable b = g.placeHolder("b", DataType.FLOAT, 1, outDim);
        SDVariable mm = g.mmul("mm", x, w);
        g.math().add("out", mm, b);
        return g;
    }

    /** Small decoder-shaped graph with position and KV placeholder inputs. */
    protected final SameDiff buildLargeDecoderGraph(int embedDim, int numLayers) {
        SameDiff g = SameDiff.create();
        SDVariable embed = g.placeHolder("inputs_embeds", DataType.FLOAT, 1, 1, embedDim);
        SDVariable posIds = g.placeHolder("position_ids", DataType.FLOAT, 1, 1);
        SDVariable x = embed.add("pos_add", posIds);

        for (int layer = 0; layer < numLayers; layer++) {
            String prefix = "layer_" + layer + "_";
            SDVariable kv = g.placeHolder(prefix + "kv", DataType.FLOAT, 1, 4, embedDim);
            SDVariable wq = g.var(prefix + "wq", Transforms.abs(
                    Nd4j.randn(DataType.FLOAT, embedDim, embedDim)).addi(0.01f));
            SDVariable wv = g.var(prefix + "wv", Transforms.abs(
                    Nd4j.randn(DataType.FLOAT, embedDim, embedDim)).addi(0.01f));

            SDVariable xFlat = g.reshape(prefix + "xflat", x, 1, embedDim);
            SDVariable q = g.mmul(prefix + "q", xFlat, wq);
            SDVariable kvMean = g.mean(prefix + "kv_mean", kv, 1);
            SDVariable kvMeanT = g.permute(prefix + "kvt", kvMean, 1, 0);
            SDVariable score = g.mmul(prefix + "score", q, kvMeanT);
            SDVariable attnOut = g.mmul(prefix + "attn_out", score,
                    g.reshape(prefix + "kvr", kvMean, 1, embedDim));
            SDVariable residual = xFlat.add(prefix + "residual", attnOut);
            x = g.reshape(prefix + "out", residual, 1, 1, embedDim);
        }

        SDVariable wFinal = g.var("w_final", Transforms.abs(
                Nd4j.randn(DataType.FLOAT, embedDim, 32)).addi(0.01f));
        SDVariable xFinal = g.reshape("x_final_flat", x, 1, embedDim);
        g.mmul("out", xFinal, wFinal);
        return g;
    }

    /** Graph with reshape boundaries between matmul gaps. */
    protected final SameDiff buildGappyGraph(int dim) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable w1 = g.var("w1", Transforms.abs(
                Nd4j.randn(DataType.FLOAT, dim, dim)).addi(0.1f));
        SDVariable w2 = g.var("w2", Transforms.abs(
                Nd4j.randn(DataType.FLOAT, dim, dim)).addi(0.1f));
        SDVariable w3 = g.var("w3", Transforms.abs(
                Nd4j.randn(DataType.FLOAT, dim, dim)).addi(0.1f));

        SDVariable mm1 = g.mmul("mm1", x, w1);
        SDVariable reshaped = g.reshape("reshape1", mm1, 1, dim);
        SDVariable mm2 = g.mmul("mm2", reshaped, w2);
        SDVariable reshaped2 = g.reshape("reshape2", mm2, 1, dim);
        g.mmul("out", reshaped2, w3);
        return g;
    }

    protected final void configureMode(SameDiff sd, GraphExecutionMode mode) {
        sd.getSessions().clear();
        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
    }

    protected final Map<String, INDArray> singlePh(String name, INDArray arr) {
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put(name, arr);
        return ph;
    }

    protected final void warmup(SameDiff sd, Map<String, INDArray> ph,
                                String outName, int steps) {
        for (int i = 0; i < steps; i++) {
            sd.output(ph, outName);
        }
    }

    protected final void warmupWithChangingInput(SameDiff sd, String phName, INDArray arr,
                                                  String outName, int steps, long[] shape) {
        Map<String, INDArray> ph = singlePh(phName, arr);
        for (int i = 0; i < steps; i++) {
            arr.assign(Nd4j.valueArrayOf(shape, i + 1.0));
            sd.output(ph, outName);
        }
    }
}
