package org.nd4j.autodiff.samediff;

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


import lombok.Data;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * VariablePattern represents a detected pattern in variable evolution during loop execution.
 * This class is used by the LoopTerminationAnalyzer to identify patterns that might indicate
 * early termination, convergence, or other significant behaviors in loop variables.
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class VariablePattern {

    /**
     * The type of pattern detected (e.g., "monotonic_decrease", "oscillation", "convergence")
     */
    private String patternType;

    /**
     * Whether this pattern is an indicator of potential loop termination
     */
    private boolean terminationIndicator;

    /**
     * Human-readable description of the pattern
     */
    private String description;

    /**
     * Confidence level of the pattern detection (0.0 to 1.0)
     */
    private double confidence = 0.0;

    /**
     * The variable name this pattern applies to
     */
    private String variableName;

    /**
     * The iteration range where this pattern was detected
     */
    private int startIteration = -1;
    private int endIteration = -1;

    /**
     * Statistical measures of the pattern
     */
    private Map<String, Double> statistics = new HashMap<>();

    /**
     * Additional metadata about the pattern
     */
    private Map<String, Object> metadata = new HashMap<>();

    /**
     * The values that led to this pattern detection
     */
    private List<Object> evidenceValues;

    /**
     * Constructor for basic pattern without confidence
     */
    public VariablePattern(String patternType, boolean terminationIndicator, String description) {
        this.patternType = patternType;
        this.terminationIndicator = terminationIndicator;
        this.description = description;
    }

    /**
     * Constructor with confidence level
     */
    public VariablePattern(String patternType, boolean terminationIndicator, String description, double confidence) {
        this.patternType = patternType;
        this.terminationIndicator = terminationIndicator;
        this.description = description;
        this.confidence = confidence;
    }

    /**
     * Constructor with full details
     */
    public VariablePattern(String patternType, boolean terminationIndicator, String description,
                           double confidence, String variableName, int startIteration, int endIteration) {
        this.patternType = patternType;
        this.terminationIndicator = terminationIndicator;
        this.description = description;
        this.confidence = confidence;
        this.variableName = variableName;
        this.startIteration = startIteration;
        this.endIteration = endIteration;
    }

    /**
     * Add a statistic about this pattern
     */
    public void addStatistic(String name, double value) {
        statistics.put(name, value);
    }

    /**
     * Add metadata about this pattern
     */
    public void addMetadata(String key, Object value) {
        metadata.put(key, value);
    }

    /**
     * Get a statistic value
     */
    public double getStatistic(String name) {
        return statistics.getOrDefault(name, 0.0);
    }

    /**
     * Check if this pattern indicates a specific behavior
     */
    public boolean indicates(String behavior) {
        return patternType.toLowerCase().contains(behavior.toLowerCase()) ||
                description.toLowerCase().contains(behavior.toLowerCase());
    }

    /**
     * Get the strength of this pattern (combination of confidence and other factors)
     */
    public double getStrength() {
        double strength = confidence;

        // Boost strength for termination indicators
        if (terminationIndicator) {
            strength += 0.2;
        }

        // Consider the iteration range - longer ranges might be more reliable
        if (startIteration >= 0 && endIteration >= 0) {
            int range = endIteration - startIteration;
            if (range > 5) {
                strength += 0.1;
            }
        }

        return Math.min(1.0, strength);
    }

    /**
     * Check if this pattern is considered strong/reliable
     */
    public boolean isStrong() {
        return getStrength() >= 0.7;
    }

    /**
     * Get a summary of this pattern
     */
    public String getSummary() {
        StringBuilder summary = new StringBuilder();
        summary.append("Pattern: ").append(patternType);
        summary.append(" (").append(String.format("%.2f", confidence)).append(" confidence)");

        if (variableName != null) {
            summary.append(" in variable '").append(variableName).append("'");
        }

        if (startIteration >= 0 && endIteration >= 0) {
            summary.append(" from iteration ").append(startIteration).append(" to ").append(endIteration);
        }

        if (terminationIndicator) {
            summary.append(" [TERMINATION INDICATOR]");
        }

        return summary.toString();
    }

    /**
     * Create a pattern for monotonic decrease
     */
    public static VariablePattern createMonotonicDecrease(String variableName, List<Object> values,
                                                          int startIter, int endIter) {
        VariablePattern pattern = new VariablePattern(
                "monotonic_decrease",
                true,
                "Variable shows consistent decrease over iterations",
                0.8,
                variableName,
                startIter,
                endIter
        );

        // Calculate statistics
        if (values != null && values.size() >= 2) {
            double totalDecrease = 0;
            int validPairs = 0;

            for (int i = 1; i < values.size(); i++) {
                Double prev = extractNumericValue(values.get(i-1));
                Double curr = extractNumericValue(values.get(i));

                if (prev != null && curr != null) {
                    totalDecrease += (prev - curr);
                    validPairs++;
                }
            }

            if (validPairs > 0) {
                pattern.addStatistic("averageDecrease", totalDecrease / validPairs);
                pattern.addStatistic("totalDecrease", totalDecrease);
                pattern.addStatistic("validPairs", validPairs);
            }
        }

        pattern.setEvidenceValues(values);
        return pattern;
    }

    /**
     * Create a pattern for convergence
     */
    public static VariablePattern createConvergence(String variableName, List<Object> values,
                                                    double convergenceRate, int startIter, int endIter) {
        VariablePattern pattern = new VariablePattern(
                "convergence",
                true,
                String.format("Variable converging at rate %.4f", convergenceRate),
                Math.min(0.95, convergenceRate * 2),
                variableName,
                startIter,
                endIter
        );

        pattern.addStatistic("convergenceRate", convergenceRate);
        pattern.addMetadata("convergenceThreshold", 0.1);
        pattern.setEvidenceValues(values);

        return pattern;
    }

    /**
     * Create a pattern for oscillation
     */
    public static VariablePattern createOscillation(String variableName, List<Object> values,
                                                    int startIter, int endIter) {
        VariablePattern pattern = new VariablePattern(
                "oscillation",
                false,
                "Variable shows oscillating behavior",
                0.6,
                variableName,
                startIter,
                endIter
        );

        // Calculate oscillation frequency
        if (values != null && values.size() >= 4) {
            int oscillations = 0;
            Double prev = extractNumericValue(values.get(0));
            boolean wasIncreasing = false;

            for (int i = 1; i < values.size(); i++) {
                Double curr = extractNumericValue(values.get(i));
                if (prev != null && curr != null) {
                    boolean isIncreasing = curr > prev;
                    if (i > 1 && isIncreasing != wasIncreasing) {
                        oscillations++;
                    }
                    wasIncreasing = isIncreasing;
                    prev = curr;
                }
            }

            pattern.addStatistic("oscillations", oscillations);
            pattern.addStatistic("oscillationFrequency", (double) oscillations / values.size());
        }

        pattern.setEvidenceValues(values);
        return pattern;
    }

    /**
     * Create a pattern for numerical instability
     */
    public static VariablePattern createNumericalInstability(String variableName, Object unstableValue,
                                                             int iteration) {
        VariablePattern pattern = new VariablePattern(
                "numerical_instability",
                true,
                "Variable became numerically unstable (NaN, Infinity, or extreme value)",
                0.9,
                variableName,
                iteration,
                iteration
        );

        pattern.addMetadata("unstableValue", unstableValue);
        pattern.addMetadata("instabilityType", classifyInstability(unstableValue));

        return pattern;
    }

    /**
     * Create a pattern for rapid change
     */
    public static VariablePattern createRapidChange(String variableName, Object oldValue, Object newValue,
                                                    int iteration) {
        VariablePattern pattern = new VariablePattern(
                "rapid_change",
                true,
                "Variable changed rapidly between iterations",
                0.7,
                variableName,
                iteration - 1,
                iteration
        );

        pattern.addMetadata("oldValue", oldValue);
        pattern.addMetadata("newValue", newValue);

        Double oldNum = extractNumericValue(oldValue);
        Double newNum = extractNumericValue(newValue);

        if (oldNum != null && newNum != null && oldNum != 0) {
            double percentChange = Math.abs((newNum - oldNum) / oldNum) * 100;
            pattern.addStatistic("percentChange", percentChange);
        }

        return pattern;
    }

    /**
     * Helper method to extract numeric value from various types
     */
    private static Double extractNumericValue(Object value) {
        if (value == null) return null;

        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        } else if (value instanceof INDArray) {
            INDArray arr = (INDArray) value;
            if (arr.isScalar()) {
                return arr.getDouble(0);
            }
        }

        return null;
    }

    /**
     * Classify the type of numerical instability
     */
    private static String classifyInstability(Object value) {
        if (value instanceof Number) {
            double d = ((Number) value).doubleValue();
            if (Double.isNaN(d)) return "NaN";
            if (Double.isInfinite(d)) return "Infinity";
            if (Math.abs(d) > 1e10) return "Extreme";
        } else if (value instanceof INDArray) {
            INDArray arr = (INDArray) value;
            if (arr.isScalar()) {
                double d = arr.getDouble(0);
                if (Double.isNaN(d)) return "NaN";
                if (Double.isInfinite(d)) return "Infinity";
                if (Math.abs(d) > 1e10) return "Extreme";
            }
        }

        return "Unknown";
    }

    @Override
    public String toString() {
        return getSummary();
    }
}