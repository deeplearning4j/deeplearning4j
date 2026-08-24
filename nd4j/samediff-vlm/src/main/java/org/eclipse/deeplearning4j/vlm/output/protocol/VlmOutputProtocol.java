/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.vlm.output.protocol;

import org.eclipse.deeplearning4j.llm.config.ModelConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;

import java.util.List;

/** Bound prompt/termination/completion/parser contract for a native VLM output grammar. */
public interface VlmOutputProtocol {
    String id();
    VlmProtocolPlan prepare(VlmProtocolRequest request, Tokenizer tokenizer, ModelConfig modelConfig);
    VlmProtocolOutput process(VlmProtocolRequest request, VlmProtocolPlan plan,
                              GenerationResult generation, Tokenizer tokenizer);
    GenerationResult mergeRegions(VlmProtocolRequest request, VlmProtocolPlan plan,
                                  List<GenerationResult> regions, Tokenizer tokenizer);
}
