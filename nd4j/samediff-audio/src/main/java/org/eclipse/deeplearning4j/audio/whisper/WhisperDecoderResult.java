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

package org.eclipse.deeplearning4j.audio.whisper;

import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * Result of Whisper transcription containing the full text,
 * detected language, and optional timestamped segments.
 */
@Data
@Builder
public class WhisperDecoderResult {

    /**
     * The full transcribed text.
     */
    private String text;

    /**
     * The detected or specified language code (e.g., "en", "fr").
     */
    private String language;

    /**
     * Timestamped segments, present when timestamp decoding is enabled.
     */
    private List<Segment> segments;

    /**
     * A timestamped segment of the transcription.
     */
    @Data
    @Builder
    public static class Segment {
        /**
         * Start time in seconds.
         */
        private double start;

        /**
         * End time in seconds.
         */
        private double end;

        /**
         * Text content of this segment.
         */
        private String text;

        /**
         * Token IDs for this segment.
         */
        private int[] tokens;
    }
}
