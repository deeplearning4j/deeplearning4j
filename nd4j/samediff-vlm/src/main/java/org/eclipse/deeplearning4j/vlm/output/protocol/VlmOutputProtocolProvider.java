/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.vlm.output.protocol;

/** Service-provider boundary for model/task-specific VLM output grammars. */
public interface VlmOutputProtocolProvider {
    String providerId();
    VlmOutputProtocol bind(String protocolId, VlmProtocolDefinition.Protocol definition);
}
