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

package org.deeplearning4j.ui.weights.beans;


import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Slightly modified version of ModelAndGradient, with binned params/gradients, suitable for fast network transfers for HistogramIterationListener
 *
 * @author Adam Gibson
 */

public class CompactModelAndGradient implements Serializable {
    private long lastUpdateTime = -1L;
    private Map<String, Map> parameters;
    private Map<String, Map> gradients;
    private double score;
    private List<Double> scores = new ArrayList<>();
    private List<Map<String, List<Double>>> updateMagnitudes = new ArrayList<>();
    private List<Map<String, List<Double>>> paramMagnitudes = new ArrayList<>();
    private List<String> layerNames = new ArrayList<>();
    private String path;


    public CompactModelAndGradient() {
        parameters = new HashMap<>();
        gradients = new HashMap<>();
    }


    public void setLastUpdateTime(long lastUpdateTime) {
        this.lastUpdateTime = lastUpdateTime;
    }

    public long getLastUpdateTime() {
        return lastUpdateTime;
    }

    public double getScore() {
        return score;
    }

    public void setScore(double score) {
        this.score = score;
    }


    public Map<String, Map> getParameters() {
        return parameters;
    }

    public void setParameters(Map<String, Map> parameters) {
        this.parameters = parameters;
    }


    public Map<String, Map> getGradients() {
        return gradients;
    }

    public void setGradients(Map<String, Map> gradients) {
        this.gradients = gradients;
    }

    public void setScores(List<Double> scores) {
        this.scores = scores;
    }

    public void setPath(String path) {
        this.path = path;
    }

    public String getPath() {
        return path;
    }

    public List<Double> getScores() {
        return scores;
    }

    public void setUpdateMagnitudes(List<Map<String, List<Double>>> updateMagnitudes) {
        this.updateMagnitudes = updateMagnitudes;
    }

    public List<Map<String, List<Double>>> getUpdateMagnitudes() {
        return updateMagnitudes;
    }

    public void setParamMagnitudes(List<Map<String, List<Double>>> paramMagnitudes) {
        this.paramMagnitudes = paramMagnitudes;
    }

    public List<Map<String, List<Double>>> getParamMagnitudes() {
        return paramMagnitudes;
    }

    public void setLayerNames(List<String> layerNames) {
        this.layerNames = layerNames;
    }

    public List<String> getLayerNames() {
        return layerNames;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        CompactModelAndGradient that = (CompactModelAndGradient) o;

        if (Double.compare(that.score, score) != 0)
            return false;
        if (parameters != null ? !parameters.equals(that.parameters) : that.parameters != null)
            return false;
        return !(gradients != null ? !gradients.equals(that.gradients) : that.gradients != null);
    }

    @Override
    public int hashCode() {
        int result;
        long temp;
        result = parameters != null ? parameters.hashCode() : 0;
        result = 31 * result + (gradients != null ? gradients.hashCode() : 0);
        temp = Double.doubleToLongBits(score);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        return result;
    }
}
