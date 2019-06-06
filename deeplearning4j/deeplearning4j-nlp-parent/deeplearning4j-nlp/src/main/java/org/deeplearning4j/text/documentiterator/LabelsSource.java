/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.text.documentiterator;

import lombok.NonNull;
import lombok.Setter;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This class is used to manage labels/documents syncronization over iterators
 *
 * @author raver119@gmail.com
 */
public class LabelsSource implements Serializable {
    private AtomicLong counter = new AtomicLong(0);
    @Setter
    private String template;
    private boolean useFormatter = false;
    private List<String> labels;
    private long maxCount = 0;
    private Set<String> uniq = Collections.newSetFromMap(new ConcurrentHashMap<String, Boolean>());

    public LabelsSource() {

    }

    /**
     * Build LabelsSource using string template.
     * Template can be raw string, in this case document counter will be appended to resulting label.
     * I.e. "SENT_" template will produce labels SENT_0, SENT_1, SENT_2 etc.
     *
     * You can also use %d formatter tag, which will be replaced with counter.
     * I.e. "SENT_%i_FLOW_1" will produce labels "SENT_0_FLOW_1", "SENT_1_FLOW_1", "SENT_2_FLOW_1" etc
     *
     * @param template
     */
    public LabelsSource(@NonNull String template) {
        this.template = template;
        if (this.template.contains("%d"))
            useFormatter = true;
    }

    public int indexOf(String label) {
        return labels.indexOf(label);
    }

    public int size() {
        return labels.size();
    }

    /**
     * Build LabelsSource using externally defined list of string labels.
     * Please note, in this case you have to make sure, the number of documents and number of labels match.
     *
     * @param labels
     */
    public LabelsSource(@NonNull List<String> labels) {
        this.labels = new ArrayList<>(labels);
        uniq.addAll(labels);
    }

    /**
     * Returns next label.
     *
     * @return next label, generated or predefined one
     */
    public synchronized String nextLabel() {
        if (labels != null) {
            return labels.get(((Long) counter.getAndIncrement()).intValue());
        } else {
            maxCount = counter.getAndIncrement();
            return formatLabel(maxCount);
        }
    }

    private String formatLabel(long value) {
        if (useFormatter)
            return String.format(template, value);
        else
            return template + value;
    }

    /**
     * This method returns the list of labels used by this generator instance.
     * If external list os labels was used as source, whole list will be returned.
     *
     * @return list of labels
     */
    public List<String> getLabels() {
        if (labels != null && !labels.isEmpty())
            return labels;
        else {
            List<String> result = new ArrayList<>();
            for (long x = 0; x < counter.get(); x++)
                result.add(formatLabel(x));
            return result;
        }
    }

    /**
     * This method is intended for storing labels retrieved from external sources.
     *
     * @param label
     */
    public void storeLabel(String label) {
        if (labels == null)
            labels = new ArrayList<>();

        if (!uniq.contains(label)) {
            uniq.add(label);
            labels.add(label);
        }
    }

    /**
     * This method should be called from Iterator's reset() method, to keep labels in sync with iterator
     */
    public void reset() {
        this.counter.set(0);
    }

    /**
     * This method returns number of labels used up to the method's call
     *
     * @return
     */
    public int getNumberOfLabelsUsed() {
        if (labels != null && !labels.isEmpty())
            return labels.size();
        else
            return ((Long) (maxCount + 1)).intValue();
    }
}
