/*
 *  * Copyright 2018 Skymind, Inc.
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
 */

package org.datavec.api.io.labels;

import org.datavec.api.writable.Writable;

import java.io.Serializable;
import java.net.URI;
import java.util.List;

/**
 * PathMultiLabelGenerator: interface to infer the label(s) of a file directly from the URI/path<br>
 * Similar to {@link PathLabelGenerator}, with 2 main differences:<br>
 * (a) Can be used for multi-label, multi-class classification (i.e., return *multiple* NDArray writables, for use in
 * networks with multiple output layers)<br>
 * (b) Does <it>not</it> support inferring label classes<br>
 * <br>
 * Regarding (b) above, this means that the implementations of PathMultiLabelGenerator typically need to (for classification
 * use cases) do one of two things (either will work, though down-stream usage of these arrays can vary slightly):
 * (a) Perform label to integer index assignment (i.e., return an IntWritable(0) for A, if you have 3 classes {A,B,C})
 * (b) Create a one-hot NDArrayWritable. For 3 classes {A,B,C} you should return a [1,0,0], [0,1,0] or [0,0,1] NDArrayWritable<br>
 * Comparatively, PathLabelGenerator can return a Text writable with the label (i.e., "class_3" or "cat") for classification.<br>
 * <br>
 * More generally, PathMultiLabelGenerator must return Writables of one of the following types:
 * {@link org.datavec.api.writable.DoubleWritable}, {@link org.datavec.api.writable.FloatWritable},
 * {@link org.datavec.api.writable.IntWritable}, {@link org.datavec.api.writable.LongWritable} or
 * {@link org.datavec.api.writable.NDArrayWritable}.<br>
 * NDArrayWritable is used for classification (via one-hot NDArrayWritable) or multi-output regression (where all values
 * are grouped together into a single array/writable) - whereas the others (double/float/int/long writables) are
 * typically used for single output regression cases, or (IntWritable) for classification where downstream classes (notably
 * DL4J's RecordReader(Multi)DataSetIterator) will convert the integer index (IntWritable) to a one-hot array ready for
 * training.<br>
 * <br>
 * In principle, you can also return time series (3d - shape [1,size,seqLength]) or images (4d - shape
 * [1,channels,height,width]) as a "label" for a given input image.
 *
 * @author Alex Black
 * @see PathLabelGenerator
 */
public interface PathMultiLabelGenerator extends Serializable {

    /**
     * @param uriPath The file or URI path to get the label for
     * @return A list of labels for the specified URI/path
     */
    List<Writable> getLabels(String uriPath);

}
