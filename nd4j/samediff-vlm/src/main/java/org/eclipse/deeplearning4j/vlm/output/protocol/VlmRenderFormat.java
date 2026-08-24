/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.vlm.output.protocol;

/** Caller-requested presentation format; independent of the model's native wire syntax. */
public enum VlmRenderFormat {
    RAW, PLAIN_TEXT, MARKDOWN, HTML, JSON
}
