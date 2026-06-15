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

package org.eclipse.deeplearning4j.vlm.eval;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.eval.EvalConfig;
import org.eclipse.deeplearning4j.llm.eval.EvalResult;
import org.eclipse.deeplearning4j.llm.eval.EvalRunner;
import org.eclipse.deeplearning4j.llm.eval.benchmark.BenchmarkTask;
import org.eclipse.deeplearning4j.llm.eval.dataset.EvalDataset;
import org.eclipse.deeplearning4j.llm.eval.dataset.EvalSample;
import org.eclipse.deeplearning4j.llm.eval.metrics.EvalMetric;
import org.eclipse.deeplearning4j.vlm.model.VisionLanguageModel;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

/**
 * Evaluation runner for Vision-Language Models.
 *
 * <p>Extends {@link EvalRunner} and supplies a VLM-specific {@link SampleGenerator}
 * that loads the image attached to each sample before calling
 * {@link VisionLanguageModel#generate}.  All aggregation, logging, and output
 * logic is handled by the shared {@link #runCoreEvaluation} loop.
 */
@Slf4j
public class VlmEvalRunner extends EvalRunner {

    /**
     * Evaluate a VLM on a benchmark task with default config.
     */
    public EvalResult evaluate(VisionLanguageModel vlm, BenchmarkTask task) throws IOException {
        return evaluate(vlm, task, EvalConfig.builder().build());
    }

    /**
     * Evaluate a VLM on a benchmark task.
     */
    public EvalResult evaluate(VisionLanguageModel vlm, BenchmarkTask task, EvalConfig config) throws IOException {
        log.info("Starting VLM evaluation: {}", task.name());
        EvalDataset dataset = task.loadDataset();
        int maxTokens = config.getMaxNewTokens() > 0 ? config.getMaxNewTokens() : task.defaultMaxNewTokens();
        int totalToEval = config.getMaxSamples() > 0 ?
                Math.min(config.getMaxSamples(), dataset.size()) : dataset.size();

        List<EvalSample> fewShotExamples = Collections.emptyList();
        SampleGenerator gen = vlmGenerator(vlm);
        return runCoreEvaluation(gen, task.name(), dataset, 0, totalToEval,
                task.primaryMetric(), task.allMetrics(),
                sample -> task.formatPrompt(sample, fewShotExamples),
                task::extractAnswer, maxTokens, config.isLogSamples(), config.getOutputFile());
    }

    /**
     * Evaluate with a custom dataset, metric, and prompt/answer functions.
     */
    public EvalResult evaluate(VisionLanguageModel vlm, EvalDataset dataset, EvalMetric metric,
                                Function<EvalSample, String> promptFormatter,
                                Function<String, String> answerExtractor) {
        return evaluate(vlm, dataset, metric, promptFormatter, answerExtractor,
                EvalConfig.builder().build());
    }

    /**
     * Evaluate with a custom dataset and full configuration.
     */
    public EvalResult evaluate(VisionLanguageModel vlm, EvalDataset dataset, EvalMetric metric,
                                Function<EvalSample, String> promptFormatter,
                                Function<String, String> answerExtractor,
                                EvalConfig config) {
        int maxTokens = config.getMaxNewTokens() > 0 ? config.getMaxNewTokens() : 256;
        int totalToEval = config.getMaxSamples() > 0 ?
                Math.min(config.getMaxSamples(), dataset.size()) : dataset.size();

        SampleGenerator gen = vlmGenerator(vlm);
        return runCoreEvaluation(gen, dataset.name(), dataset, 0, totalToEval,
                metric, List.of(metric), promptFormatter, answerExtractor,
                maxTokens, config.isLogSamples(), config.getOutputFile());
    }

    /**
     * Builds a {@link SampleGenerator} that loads the image for each sample and
     * delegates to {@link VisionLanguageModel#generate}.  Samples with no
     * resolvable image path produce an empty string so that the evaluation loop
     * can continue rather than abort.
     */
    private SampleGenerator vlmGenerator(VisionLanguageModel vlm) {
        return (sample, prompt, maxTokens) -> {
            INDArray image = loadImage(sample.getImagePath());
            if (image == null) {
                log.warn("No image for sample {}, skipping", sample.getId());
                return "";
            }
            return vlm.generate(image, prompt, maxTokens, 0.0, false);
        };
    }

    private INDArray loadImage(String imagePath) {
        if (imagePath == null || imagePath.isEmpty()) return null;
        try {
            File imageFile = new File(imagePath);
            if (!imageFile.exists()) {
                log.warn("Image file not found: {}", imagePath);
                return null;
            }
            BufferedImage img = ImageIO.read(imageFile);
            if (img == null) return null;

            int h = img.getHeight();
            int w = img.getWidth();
            float[] pixels = new float[3 * h * w];
            int idx = 0;
            for (int c = 0; c < 3; c++) {
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        int rgb = img.getRGB(x, y);
                        switch (c) {
                            case 0: pixels[idx++] = ((rgb >> 16) & 0xFF) / 255.0f; break;
                            case 1: pixels[idx++] = ((rgb >> 8) & 0xFF) / 255.0f; break;
                            case 2: pixels[idx++] = (rgb & 0xFF) / 255.0f; break;
                        }
                    }
                }
            }
            return Nd4j.createFromArray(pixels).reshape(1, 3, h, w);
        } catch (IOException e) {
            log.warn("Failed to load image {}: {}", imagePath, e.getMessage());
            return null;
        }
    }
}
