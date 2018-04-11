/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.eval;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import lombok.Getter;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class ConfusionMatrix<T extends Comparable<? super T>> implements Serializable {
    @Getter
    private volatile Map<T, Multiset<T>> matrix;
    private List<T> classes;

    /**
     * Creates an empty confusion Matrix
     */
    public ConfusionMatrix(List<T> classes) {
        this.matrix = new ConcurrentHashMap<>();
        this.classes = classes;
    }

    public ConfusionMatrix() {
        this(new ArrayList<T>());
    }

    /**
     * Creates a new ConfusionMatrix initialized with the contents of another ConfusionMatrix.
     */
    public ConfusionMatrix(ConfusionMatrix<T> other) {
        this(other.getClasses());
        this.add(other);
    }

    /**
     * Increments the entry specified by actual and predicted by one.
     */
    public synchronized void add(T actual, T predicted) {
        add(actual, predicted, 1);
    }

    /**
     * Increments the entry specified by actual and predicted by count.
     */
    public synchronized void add(T actual, T predicted, int count) {
        if (matrix.containsKey(actual)) {
            matrix.get(actual).add(predicted, count);
        } else {
            Multiset<T> counts = HashMultiset.create();
            counts.add(predicted, count);
            matrix.put(actual, counts);
        }
    }

    /**
     * Adds the entries from another confusion matrix to this one.
     */
    public synchronized void add(ConfusionMatrix<T> other) {
        for (T actual : other.matrix.keySet()) {
            Multiset<T> counts = other.matrix.get(actual);
            for (T predicted : counts.elementSet()) {
                int count = counts.count(predicted);
                this.add(actual, predicted, count);
            }
        }
    }

    /**
     * Gives the applyTransformToDestination of all classes in the confusion matrix.
     */
    public List<T> getClasses() {
        if (classes == null)
            classes = new ArrayList<>();
        return classes;
    }

    /**
     * Gives the count of the number of times the "predicted" class was predicted for the "actual"
     * class.
     */
    public synchronized int getCount(T actual, T predicted) {
        if (!matrix.containsKey(actual)) {
            return 0;
        } else {
            return matrix.get(actual).count(predicted);
        }
    }

    /**
     * Computes the total number of times the class was predicted by the classifier.
     */
    public synchronized int getPredictedTotal(T predicted) {
        int total = 0;
        for (T actual : classes) {
            total += getCount(actual, predicted);
        }
        return total;
    }

    /**
     * Computes the total number of times the class actually appeared in the data.
     */
    public synchronized int getActualTotal(T actual) {
        if (!matrix.containsKey(actual)) {
            return 0;
        } else {
            int total = 0;
            for (T elem : matrix.get(actual).elementSet()) {
                total += matrix.get(actual).count(elem);
            }
            return total;
        }
    }

    @Override
    public String toString() {
        return matrix.toString();
    }

    /**
     * Outputs the ConfusionMatrix as comma-separated values for easy import into spreadsheets
     */
    public String toCSV() {
        StringBuilder builder = new StringBuilder();

        // Header Row
        builder.append(",,Predicted Class,\n");

        // Predicted Classes Header Row
        builder.append(",,");
        for (T predicted : classes) {
            builder.append(String.format("%s,", predicted));
        }
        builder.append("Total\n");

        // Data Rows
        String firstColumnLabel = "Actual Class,";
        for (T actual : classes) {
            builder.append(firstColumnLabel);
            firstColumnLabel = ",";
            builder.append(String.format("%s,", actual));

            for (T predicted : classes) {
                builder.append(getCount(actual, predicted));
                builder.append(",");
            }
            // Actual Class Totals Column
            builder.append(getActualTotal(actual));
            builder.append("\n");
        }

        // Predicted Class Totals Row
        builder.append(",Total,");
        for (T predicted : classes) {
            builder.append(getPredictedTotal(predicted));
            builder.append(",");
        }
        builder.append("\n");

        return builder.toString();
    }

    /**
     * Outputs Confusion Matrix in an HTML table. Cascading Style Sheets (CSS) can control the table's
     * appearance by defining the empty-space, actual-count-header, predicted-class-header, and
     * count-element classes. For example
     *
     * @return html string
     */
    public String toHTML() {
        StringBuilder builder = new StringBuilder();

        int numClasses = classes.size();
        // Header Row
        builder.append("<table>\n");
        builder.append("<tr><th class=\"empty-space\" colspan=\"2\" rowspan=\"2\">");
        builder.append(String.format("<th class=\"predicted-class-header\" colspan=\"%d\">Predicted Class</th></tr>%n",
                        numClasses + 1));

        // Predicted Classes Header Row
        builder.append("<tr>");
        // builder.append("<th></th><th></th>");
        for (T predicted : classes) {
            builder.append("<th class=\"predicted-class-header\">");
            builder.append(predicted);
            builder.append("</th>");
        }
        builder.append("<th class=\"predicted-class-header\">Total</th>");
        builder.append("</tr>\n");

        // Data Rows
        String firstColumnLabel = String.format(
                        "<tr><th class=\"actual-class-header\" rowspan=\"%d\">Actual Class</th>", numClasses + 1);
        for (T actual : classes) {
            builder.append(firstColumnLabel);
            firstColumnLabel = "<tr>";
            builder.append(String.format("<th class=\"actual-class-header\" >%s</th>", actual));

            for (T predicted : classes) {
                builder.append("<td class=\"count-element\">");
                builder.append(getCount(actual, predicted));
                builder.append("</td>");
            }

            // Actual Class Totals Column
            builder.append("<td class=\"count-element\">");
            builder.append(getActualTotal(actual));
            builder.append("</td>");
            builder.append("</tr>\n");
        }

        // Predicted Class Totals Row
        builder.append("<tr><th class=\"actual-class-header\">Total</th>");
        for (T predicted : classes) {
            builder.append("<td class=\"count-element\">");
            builder.append(getPredictedTotal(predicted));
            builder.append("</td>");
        }
        builder.append("<td class=\"empty-space\"></td>\n");
        builder.append("</tr>\n");
        builder.append("</table>\n");

        return builder.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof ConfusionMatrix))
            return false;
        ConfusionMatrix<?> c = (ConfusionMatrix<?>) o;
        return matrix.equals(c.matrix) && classes.equals(c.classes);
    }

    @Override
    public int hashCode() {
        int result = 17;
        result = 31 * result + (matrix == null ? 0 : matrix.hashCode());
        result = 31 * result + (classes == null ? 0 : classes.hashCode());
        return result;
    }
}
