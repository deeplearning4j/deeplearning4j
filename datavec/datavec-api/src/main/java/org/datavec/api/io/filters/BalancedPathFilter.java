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

package org.datavec.api.io.filters;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.writable.Writable;

import java.net.URI;
import java.util.*;

/**
 * Randomizes the order of paths in an array and removes paths randomly
 * to have the same number of paths for each label. Further interlaces the paths
 * on output based on their labels, to obtain easily optimal batches for training.
 *
 * @author saudet
 */
public class BalancedPathFilter extends RandomPathFilter {

    protected PathLabelGenerator labelGenerator;
    protected long maxLabels = 0, minPathsPerLabel = 0, maxPathsPerLabel = 0;
    protected String[] labels = null;

    /** Calls {@code this(random, extensions, labelGenerator, 0, 0, 0, 0)}. */
    public BalancedPathFilter(Random random, String[] extensions, PathLabelGenerator labelGenerator) {
        this(random, extensions, labelGenerator, 0, 0, 0, 0);
    }

    /** Calls {@code this(random, null, labelGenerator, 0, 0, 0, maxPathsPerLabel)}. */
    public BalancedPathFilter(Random random, PathLabelGenerator labelGenerator, long maxPathsPerLabel) {
        this(random, null, labelGenerator, 0, 0, 0, maxPathsPerLabel);
    }

    /** Calls {@code this(random, extensions, labelGenerator, 0, 0, 0, maxPathsPerLabel)}. */
    public BalancedPathFilter(Random random, String[] extensions, PathLabelGenerator labelGenerator,
                    long maxPathsPerLabel) {
        this(random, extensions, labelGenerator, 0, 0, 0, maxPathsPerLabel);
    }

    /** Calls {@code this(random, extensions, labelGenerator, 0, maxLabels, 0, maxPathsPerLabel)}. */
    public BalancedPathFilter(Random random, PathLabelGenerator labelGenerator, long maxPaths, long maxLabels,
                    long maxPathsPerLabel) {
        this(random, null, labelGenerator, maxPaths, maxLabels, 0, maxPathsPerLabel);
    }

    /** Calls {@code this(random, extensions, labelGenerator, 0, maxLabels, 0, maxPathsPerLabel)}. */
    public BalancedPathFilter(Random random, String[] extensions, PathLabelGenerator labelGenerator, long maxLabels,
                    long maxPathsPerLabel) {
        this(random, extensions, labelGenerator, 0, maxLabels, 0, maxPathsPerLabel);
    }

    /**
     * Constructs an instance of the PathFilter. If {@code minPathsPerLabel > 0},
     * it might return an unbalanced set if the value is larger than the number of
     * examples available for the label with the minimum amount.
     *
     * @param random           object to use
     * @param extensions       of files to keep
     * @param labelGenerator   to obtain labels from paths
     * @param maxPaths         max number of paths to return (0 == unlimited)
     * @param maxLabels        max number of labels to return (0 == unlimited)
     * @param minPathsPerLabel min number of paths per labels to return
     * @param maxPathsPerLabel max number of paths per labels to return (0 == unlimited)
     * @param labels           of the paths to keep (empty set == keep all paths)
     */
    public BalancedPathFilter(Random random, String[] extensions, PathLabelGenerator labelGenerator, long maxPaths,
                    long maxLabels, long minPathsPerLabel, long maxPathsPerLabel, String... labels) {
        super(random, extensions, maxPaths);
        this.labelGenerator = labelGenerator;
        this.maxLabels = maxLabels;
        this.minPathsPerLabel = minPathsPerLabel;
        this.maxPathsPerLabel = maxPathsPerLabel;
        this.labels = labels;
    }

    protected boolean acceptLabel(String name) {
        if (labels == null || labels.length == 0) {
            return true;
        }
        for (String label : labels) {
            if (name.equals(label)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public URI[] filter(URI[] paths) {
        paths = super.filter(paths);
        if (labelGenerator == null)
            labelGenerator = new ParentPathLabelGenerator();
        Map<Writable, List<URI>> labelPaths = new LinkedHashMap<Writable, List<URI>>();
        for (int i = 0; i < paths.length; i++) {
            URI path = paths[i];
            Writable label = labelGenerator.getLabelForPath(path);
            if (!acceptLabel(label.toString())) {
                continue;
            }
            List<URI> pathList = labelPaths.get(label);
            if (pathList == null) {
                if (maxLabels > 0 && labelPaths.size() >= maxLabels) {
                    continue;
                }
                labelPaths.put(label, pathList = new ArrayList<URI>());
            }
            pathList.add(path);
        }

        int minCount = maxPathsPerLabel > 0 ?
                (int)Math.min(maxPathsPerLabel, Integer.MAX_VALUE) : Integer.MAX_VALUE;
        for (List<URI> pathList : labelPaths.values()) {
            if (minCount > pathList.size()) {
                minCount = pathList.size();
            }
        }
        if (minCount < minPathsPerLabel) {
            minCount = (int)Math.min(minPathsPerLabel, Integer.MAX_VALUE);
        }

        ArrayList<URI> newpaths = new ArrayList<URI>();
        for (int i = 0; i < minCount; i++) {
            for (List<URI> p : labelPaths.values()) {
                if (i < p.size()) {
                    newpaths.add(p.get(i));
                }
            }
        }
        return newpaths.toArray(new URI[newpaths.size()]);
    }
}
