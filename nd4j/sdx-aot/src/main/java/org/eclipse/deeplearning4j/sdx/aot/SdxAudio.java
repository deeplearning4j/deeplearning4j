/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.eclipse.deeplearning4j.sdx.aot;

import org.eclipse.deeplearning4j.audio.whisper.WhisperDecoderResult;
import org.eclipse.deeplearning4j.audio.whisper.WhisperModel;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.File;

/**
 * Audio (Whisper speech-to-text) facade for the SDX AOT exports.
 *
 * <p>Contract consumed by {@link SdxLlmCli} ({@code transcribe} subcommand) and
 * {@link SdxLlmCApi} ({@code sdxAudioTranscribe}): keep the signature stable.</p>
 *
 * <p>Model path resolution order:
 * <ol>
 *   <li>If the path is a directory containing {@code encoder_model.onnx} — ONNX model directory.</li>
 *   <li>If the path ends with {@code .bin} or {@code .ggml} or {@code .gguf} — GGML/GGUF binary.</li>
 *   <li>Otherwise attempt ONNX directory load (lets WhisperModel raise a clear error).</li>
 * </ol>
 * </p>
 *
 * <p>Options JSON keys (all optional):
 * <ul>
 *   <li>{@code language} — ISO 639-1 language code, e.g. {@code "en"}; {@code null} / absent = auto-detect.</li>
 *   <li>{@code task} — {@code "transcribe"} (default) or {@code "translate"}.</li>
 *   <li>{@code timestamps} — boolean; {@code false} by default.</li>
 *   <li>{@code maxNewTokens} — integer max decode tokens; default is the WhisperModel default (448).</li>
 * </ul>
 * </p>
 */
final class SdxAudio {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private SdxAudio() {
    }

    /**
     * Transcribe an audio file with a Whisper-family model.
     *
     * @param modelPath   model file or directory (ONNX directory or GGML/GGUF .bin file)
     * @param audioPath   input audio (WAV; other containers if the Java Sound SPI supports them)
     * @param optionsJson optional JSON: {@code {"language":"en","task":"transcribe","maxNewTokens":N}}
     * @return the transcript text (trimmed)
     * @throws Exception on load failure, missing files, or transcription errors
     */
    static String transcribe(String modelPath, String audioPath, String optionsJson) throws Exception {
        // Parse options — shaded Jackson tree API only (no databind/readValue with POJOs).
        JsonNode opts = (optionsJson == null || optionsJson.trim().isEmpty())
                ? MAPPER.createObjectNode()
                : MAPPER.readTree(optionsJson);

        String language = opts.path("language").isMissingNode() || opts.path("language").isNull()
                ? "en" : opts.path("language").asText("en");
        String task = opts.path("task").asText("transcribe");
        boolean timestamps = opts.path("timestamps").asBoolean(false);

        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            throw new IllegalArgumentException("Model path does not exist: " + modelPath);
        }
        File audioFile = new File(audioPath);
        if (!audioFile.exists()) {
            throw new IllegalArgumentException("Audio file does not exist: " + audioPath);
        }

        try (WhisperModel model = loadModel(modelFile, opts)) {
            WhisperDecoderResult result = model.transcribe(audioFile, language, task, timestamps);
            String text = result.getText();
            return text != null ? text.trim() : "";
        }
    }

    /**
     * Load a WhisperModel from either an ONNX directory or a GGML/GGUF binary.
     *
     * <p>Format detection:
     * <ul>
     *   <li>Directory containing {@code encoder_model.onnx} → ONNX path.</li>
     *   <li>Regular file with extension {@code .bin}, {@code .ggml}, or {@code .gguf} → GGML path.</li>
     *   <li>Directory without {@code encoder_model.onnx} → ONNX path (WhisperModel raises the error).</li>
     * </ul>
     * </p>
     */
    private static WhisperModel loadModel(File modelFile, JsonNode opts) throws Exception {
        if (modelFile.isDirectory()) {
            return WhisperModel.fromOnnx(modelFile);
        }
        // Regular file: sniff by extension.
        String name = modelFile.getName().toLowerCase(java.util.Locale.ROOT);
        if (name.endsWith(".bin") || name.endsWith(".ggml") || name.endsWith(".gguf")) {
            return WhisperModel.fromGgml(modelFile);
        }
        // Unknown extension — try ONNX directory anyway (e.g., symlink pointing to dir).
        return WhisperModel.fromOnnx(modelFile);
    }
}
