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

package org.eclipse.deeplearning4j.llm.eval;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.eval.benchmark.BenchmarkTask;
import org.eclipse.deeplearning4j.llm.eval.dataset.EvalDataset;
import org.eclipse.deeplearning4j.llm.eval.dataset.EvalSample;
import org.eclipse.deeplearning4j.llm.eval.metrics.EvalMetric;
import org.eclipse.deeplearning4j.llm.generation.TextGenerator;

import java.io.IOException;
import java.util.*;
import java.util.function.Function;

/**
 * Orchestrates evaluation of LLM models against benchmarks and custom datasets.
 */
@Slf4j
public class EvalRunner {

    public EvalResult evaluate(TextGenerator generator, BenchmarkTask task) throws IOException {
        return evaluate(generator, task, EvalConfig.builder().build());
    }

    public EvalResult evaluate(TextGenerator generator, BenchmarkTask task, EvalConfig config) throws IOException {
        log.info("Starting evaluation: {} (fewShot={}, maxSamples={})",
                task.name(),
                config.getNumFewShot() < 0 ? task.defaultFewShot() : config.getNumFewShot(),
                config.getMaxSamples());

        EvalDataset dataset = task.loadDataset();
        int numFewShot = config.getNumFewShot() < 0 ? task.defaultFewShot() : config.getNumFewShot();
        int maxTokens = config.getMaxNewTokens() > 0 ? config.getMaxNewTokens() : task.defaultMaxNewTokens();

        List<EvalSample> fewShotExamples = new ArrayList<>();
        int evalStart = 0;
        if (numFewShot > 0 && dataset.size() > numFewShot) {
            for (int i = 0; i < numFewShot && i < dataset.size(); i++) {
                fewShotExamples.add(dataset.get(i));
            }
            evalStart = numFewShot;
        }

        int totalToEval = dataset.size() - evalStart;
        if (config.getMaxSamples() > 0) {
            totalToEval = Math.min(totalToEval, config.getMaxSamples());
        }

        return runEvaluation(generator, task.name(), dataset, evalStart, totalToEval,
                fewShotExamples, task.primaryMetric(), task.allMetrics(),
                sample -> task.formatPrompt(sample, fewShotExamples),
                task::extractAnswer, maxTokens, config.isLogSamples(), config.getOutputFile());
    }

    public EvalResult evaluate(TextGenerator generator, EvalDataset dataset, EvalMetric metric,
                                Function<EvalSample, String> promptFormatter,
                                Function<String, String> answerExtractor) {
        return evaluate(generator, dataset, metric, promptFormatter, answerExtractor,
                EvalConfig.builder().build());
    }

    public EvalResult evaluate(TextGenerator generator, EvalDataset dataset, EvalMetric metric,
                                Function<EvalSample, String> promptFormatter,
                                Function<String, String> answerExtractor,
                                EvalConfig config) {
        int maxTokens = config.getMaxNewTokens() > 0 ? config.getMaxNewTokens() : 256;
        int totalToEval = config.getMaxSamples() > 0 ?
                Math.min(config.getMaxSamples(), dataset.size()) : dataset.size();

        return runEvaluation(generator, dataset.name(), dataset, 0, totalToEval,
                Collections.emptyList(), metric, List.of(metric),
                promptFormatter, answerExtractor, maxTokens,
                config.isLogSamples(), config.getOutputFile());
    }

    public Map<String, EvalResult> evaluateAll(TextGenerator generator, List<BenchmarkTask> tasks) throws IOException {
        return evaluateAll(generator, tasks, EvalConfig.builder().build());
    }

    public Map<String, EvalResult> evaluateAll(TextGenerator generator, List<BenchmarkTask> tasks,
                                                 EvalConfig config) throws IOException {
        Map<String, EvalResult> results = new LinkedHashMap<>();
        for (BenchmarkTask task : tasks) {
            log.info("=== Evaluating: {} ===", task.name());
            EvalResult result = evaluate(generator, task, config);
            results.put(task.name(), result);
            log.info("Result: {} = {:.4f}", task.name(), result.getPrimaryScore());
        }
        return results;
    }

    private EvalResult runEvaluation(TextGenerator generator, String benchmarkName,
                                      EvalDataset dataset, int startIdx, int count,
                                      List<EvalSample> fewShotExamples,
                                      EvalMetric primaryMetric, List<EvalMetric> allMetrics,
                                      Function<EvalSample, String> promptFormatter,
                                      Function<String, String> answerExtractor,
                                      int maxTokens, boolean logSamples,
                                      java.io.File outputFile) {
        long startTime = System.currentTimeMillis();
        List<SampleResult> sampleResults = new ArrayList<>();
        List<Double> primaryScores = new ArrayList<>();
        Map<String, List<Double>> allScores = new LinkedHashMap<>();
        Map<String, List<Double>> categoryScores = new LinkedHashMap<>();

        for (EvalMetric m : allMetrics) {
            allScores.put(m.name(), new ArrayList<>());
        }

        int correctCount = 0;
        int evaluated = 0;

        for (int i = startIdx; i < startIdx + count && i < dataset.size(); i++) {
            EvalSample sample = dataset.get(i);
            String prompt = promptFormatter.apply(sample);

            String rawOutput;
            try {
                rawOutput = generator.generate(prompt, maxTokens);
            } catch (Exception e) {
                log.warn("Generation failed for sample {}: {}", sample.getId(), e.getMessage());
                rawOutput = "";
            }

            String prediction = answerExtractor.apply(rawOutput);

            double primaryScore = primaryMetric.score(prediction, sample.getReferences());
            primaryScores.add(primaryScore);
            boolean correct = primaryScore >= 1.0;
            if (correct) correctCount++;

            for (EvalMetric m : allMetrics) {
                double score = m.score(prediction, sample.getReferences());
                allScores.get(m.name()).add(score);
            }

            if (sample.getSubject() != null) {
                categoryScores.computeIfAbsent(sample.getSubject(), k -> new ArrayList<>())
                        .add(primaryScore);
            }

            if (logSamples) {
                sampleResults.add(SampleResult.builder()
                        .sampleId(sample.getId())
                        .prediction(prediction)
                        .references(sample.getReferences())
                        .score(primaryScore)
                        .correct(correct)
                        .build());
            }

            evaluated++;
            if (evaluated % 100 == 0) {
                double runningScore = primaryMetric.aggregate(primaryScores);
                log.info("Progress: {}/{} samples, running score: {:.4f}", evaluated, count, runningScore);
            }
        }

        long evalTimeMs = System.currentTimeMillis() - startTime;

        Map<String, Double> aggregatedMetrics = new LinkedHashMap<>();
        for (EvalMetric m : allMetrics) {
            aggregatedMetrics.put(m.name(), m.aggregate(allScores.get(m.name())));
        }

        Map<String, Double> aggregatedCategories = new LinkedHashMap<>();
        for (Map.Entry<String, List<Double>> entry : categoryScores.entrySet()) {
            aggregatedCategories.put(entry.getKey(), primaryMetric.aggregate(entry.getValue()));
        }

        EvalResult result = EvalResult.builder()
                .benchmarkName(benchmarkName)
                .primaryScore(primaryMetric.aggregate(primaryScores))
                .metricScores(aggregatedMetrics)
                .categoryScores(aggregatedCategories.isEmpty() ? null : aggregatedCategories)
                .totalSamples(evaluated)
                .correctSamples(correctCount)
                .evaluationTimeMs(evalTimeMs)
                .sampleResults(logSamples ? sampleResults : null)
                .build();

        log.info("Evaluation complete: {} - score={:.4f}, {}/{} correct, {}ms",
                benchmarkName, result.getPrimaryScore(), correctCount, evaluated, evalTimeMs);

        if (outputFile != null) {
            try {
                result.writeJson(outputFile);
                log.info("Results written to {}", outputFile);
            } catch (Exception e) {
                log.warn("Failed to write results to {}: {}", outputFile, e.getMessage());
            }
        }

        return result;
    }
}
