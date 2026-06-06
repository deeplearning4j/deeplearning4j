/*
 *  ******************************************************************************
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

package org.eclipse.deeplearning4j.llm.pipeline.resolution;

import lombok.Getter;
import lombok.Setter;

import java.util.*;

/**
 * A canonical resolved entity that aggregates multiple mentions.
 * Tracks the canonical name, type, all surface-form mentions,
 * and the confidence that each mention belongs to this entity.
 */
@Getter
public class ResolvedEntity {
    private final String id;
    @Setter
    private String canonicalName;
    @Setter
    private String type;
    private final List<EntityMention> mentions;
    @Setter
    private boolean manuallyVerified;

    public ResolvedEntity(String id, String canonicalName, String type) {
        this.id = id;
        this.canonicalName = canonicalName;
        this.type = type;
        this.mentions = new ArrayList<>();
        this.manuallyVerified = false;
    }

    public void addMention(EntityMention mention) {
        mentions.add(mention);
    }

    public int mentionCount() {
        return mentions.size();
    }

    /**
     * Average confidence across all mentions.
     */
    public double averageConfidence() {
        if (mentions.isEmpty()) return 0.0;
        double sum = 0.0;
        for (EntityMention m : mentions) {
            sum += m.getConfidence();
        }
        return sum / mentions.size();
    }

    /**
     * The highest confidence among all mentions.
     */
    public double maxConfidence() {
        double max = 0.0;
        for (EntityMention m : mentions) {
            if (m.getConfidence() > max) max = m.getConfidence();
        }
        return max;
    }

    /**
     * All unique surface forms seen for this entity.
     */
    public Set<String> surfaceForms() {
        Set<String> forms = new LinkedHashSet<>();
        for (EntityMention m : mentions) {
            forms.add(m.getText());
        }
        return forms;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        return id.equals(((ResolvedEntity) o).id);
    }

    @Override
    public int hashCode() {
        return id.hashCode();
    }

    @Override
    public String toString() {
        return "ResolvedEntity{" + type + ": " + canonicalName
                + " (" + mentions.size() + " mentions, avg="
                + String.format("%.2f", averageConfidence()) + ")}";
    }
}
