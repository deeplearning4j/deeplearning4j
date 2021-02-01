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

package org.deeplearning4j.spark.parameterserver.python;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;


public class DataSetDescriptor implements java.io.Serializable{
    private ArrayDescriptor features, labels;
    private ArrayDescriptor featuresMask;
    private ArrayDescriptor labelsMask;
    private boolean preProcessed;

    public DataSetDescriptor(ArrayDescriptor features, ArrayDescriptor labels, ArrayDescriptor featuresMask, ArrayDescriptor labelsMask){
        this.features = features;
        this.labels = labels;
        this.featuresMask = featuresMask;
        this.labelsMask = labelsMask;
    }

    public DataSetDescriptor(DataSet ds)throws Exception{
        features = new ArrayDescriptor(ds.getFeatures());
        labels = new ArrayDescriptor(ds.getLabels());
        INDArray featuresMask = ds.getFeaturesMaskArray();
        if (featuresMask == null){
            this.featuresMask = null;
        }
        else{
            this.featuresMask = new ArrayDescriptor(featuresMask);
        }
        INDArray labelsMask = ds.getLabelsMaskArray();
        if (labelsMask == null){
            this.labelsMask = null;
        }
        else{
            this.labelsMask = new ArrayDescriptor(labelsMask);
        }

        preProcessed = ds.isPreProcessed();
    }

    public DataSet getDataSet(){
        INDArray features = this.features.getArray();
        INDArray labels = this.labels.getArray();
        INDArray featuresMask;
        INDArray labelsMask;
        if (this.featuresMask == null){
            featuresMask = null;
        }
        else{
            featuresMask = this.featuresMask.getArray();
        }
        if (this.labelsMask == null){
            labelsMask = null;
        }
        else{
            labelsMask = this.labelsMask.getArray();
        }
        DataSet ds = new DataSet(features, labels, featuresMask, labelsMask);
        if(preProcessed) {
            ds.markAsPreProcessed();
        }
        return ds;
    }

    public ArrayDescriptor getFeatures() {
        return features;
    }

    public ArrayDescriptor getLabels() {
        return labels;
    }

    public ArrayDescriptor getFeaturesMask() {
        return featuresMask;
    }

    public ArrayDescriptor getLabelsMask() {
        return labelsMask;
    }
}
