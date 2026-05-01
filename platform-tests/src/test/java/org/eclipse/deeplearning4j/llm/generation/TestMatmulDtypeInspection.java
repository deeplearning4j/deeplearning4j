package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.buffer.DataType;

import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;

import java.util.*;

/**
 * Diagnostic: dump dtype of every matmul op's inputs in SmolDocling decoder,
 * both pre- and post-GraphOptimizer.
 * Run with:
 *   cd platform-tests && mvn test -Dtest=TestMatmulDtypeInspection -Dbackend.artifactId=nd4j-cuda-12.9 2>&1 | tee /tmp/matmul-dtype.log
 */
@Slf4j
public class TestMatmulDtypeInspection {

    @Test
    public void inspectMatmulInputDtypes() throws Exception {
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedTokensResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
                embedTokensResult.getModelFile().getAbsolutePath()
        );
        SameDiff rawDecoder = models[1];

        // Dump PRE-optimizer state
        log.info("########## PRE-OPTIMIZER ##########");
        dumpDtypeInfo(rawDecoder);

        // Run optimizer
        log.info("Running GraphOptimizer...");
        List<String> outputs = new ArrayList<>(rawDecoder.outputs());
        SameDiff decoder = GraphOptimizer.optimize(rawDecoder, outputs);
        int opsBefore = rawDecoder.getOps().size();
        int opsAfter = decoder.getOps().size();
        log.info("GraphOptimizer: {} -> {} ops ({} removed)", opsBefore, opsAfter, opsBefore - opsAfter);

        // Dump POST-optimizer state
        log.info("########## POST-OPTIMIZER ##########");
        dumpDtypeInfo(decoder);

    }

    private void dumpDtypeInfo(SameDiff sd) {
        // Count matmul ops by input dtype combination
        Map<String, Integer> dtypeComboCounts = new TreeMap<>();
        Map<String, List<String>> dtypeComboExamples = new TreeMap<>();
        int totalMatmuls = 0;

        for (SameDiffOp op : sd.getOps().values()) {
            String opName = op.getOp().opName();
            if (!"matmul".equals(opName) && !"mmul".equals(opName)) continue;
            totalMatmuls++;

            List<String> inputs = op.getInputsToOp();
            StringBuilder dtypes = new StringBuilder();
            for (int i = 0; i < inputs.size(); i++) {
                SDVariable var = sd.getVariable(inputs.get(i));
                DataType dt = var != null ? var.dataType() : DataType.UNKNOWN;
                if (i > 0) dtypes.append(" x ");
                dtypes.append(dt);
            }
            String combo = dtypes.toString();
            dtypeComboCounts.merge(combo, 1, Integer::sum);
            dtypeComboExamples.computeIfAbsent(combo, k -> new ArrayList<>());
            if (dtypeComboExamples.get(combo).size() < 3) {
                dtypeComboExamples.get(combo).add(op.getName());
            }
        }

        log.info("=== MATMUL INPUT DTYPE SUMMARY ({} total matmul ops) ===", totalMatmuls);
        for (var entry : dtypeComboCounts.entrySet()) {
            log.info("  {} : {} ops  (examples: {})",
                    entry.getKey(), entry.getValue(),
                    dtypeComboExamples.get(entry.getKey()));
        }

        // Cast ops: what direction?
        Map<String, Integer> castDirections = new TreeMap<>();
        int totalCasts = 0;
        for (SameDiffOp op : sd.getOps().values()) {
            if (!"cast".equals(op.getOp().opName())) continue;
            totalCasts++;
            List<String> inputs = op.getInputsToOp();
            List<String> outputs = op.getOutputsOfOp();
            SDVariable inVar = sd.getVariable(inputs.get(0));
            SDVariable outVar = sd.getVariable(outputs.get(0));
            DataType inDt = inVar != null ? inVar.dataType() : DataType.UNKNOWN;
            DataType outDt = outVar != null ? outVar.dataType() : DataType.UNKNOWN;
            String direction = inDt + " -> " + outDt;
            castDirections.merge(direction, 1, Integer::sum);
        }

        log.info("=== CAST OP DIRECTION SUMMARY ({} total cast ops) ===", totalCasts);
        for (var entry : castDirections.entrySet()) {
            log.info("  {} : {} ops", entry.getKey(), entry.getValue());
        }

        // Op type histogram
        Map<String, Integer> opHist = new TreeMap<>();
        for (SameDiffOp op : sd.getOps().values()) {
            opHist.merge(op.getOp().opName(), 1, Integer::sum);
        }
        log.info("=== OP HISTOGRAM ({} total ops) ===", sd.getOps().size());
        log.info("  {}", opHist);

        // FP32 matmul input provenance (first 10)
        log.info("=== FP32 MATMUL INPUT PROVENANCE ===");
        int fp32Count = 0;
        for (SameDiffOp op : sd.getOps().values()) {
            String opName = op.getOp().opName();
            if (!"matmul".equals(opName) && !"mmul".equals(opName)) continue;

            List<String> inputs = op.getInputsToOp();
            for (int i = 0; i < inputs.size(); i++) {
                SDVariable var = sd.getVariable(inputs.get(i));
                if (var != null && var.dataType() == DataType.FLOAT) {
                    String producerOp = "NONE (placeholder/constant)";
                    for (SameDiffOp candidate : sd.getOps().values()) {
                        if (candidate.getOutputsOfOp() != null &&
                                candidate.getOutputsOfOp().contains(inputs.get(i))) {
                            producerOp = candidate.getOp().opName() + " (" + candidate.getName() + ")";
                            break;
                        }
                    }
                    if (fp32Count < 10) {
                        log.info("  matmul '{}' input[{}] '{}' is {}, produced by: {}",
                                op.getName(), i, inputs.get(i), var.dataType(), producerOp);
                    }
                    fp32Count++;
                }
            }
        }
        log.info("Total FP32 matmul inputs: {}", fp32Count);

        // Also check HALF matmul inputs
        int halfCount = 0;
        for (SameDiffOp op : sd.getOps().values()) {
            String opName = op.getOp().opName();
            if (!"matmul".equals(opName) && !"mmul".equals(opName)) continue;
            List<String> inputs = op.getInputsToOp();
            for (String input : inputs) {
                SDVariable var = sd.getVariable(input);
                if (var != null && var.dataType() == DataType.HALF) halfCount++;
            }
        }
        log.info("Total HALF matmul inputs: {}", halfCount);
    }
}
