/*
 *  * Copyright 2017 Skymind, Inc.
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
 *     networks with multiple output layers)<br>
 * (b) Does <it>not</it> support inferring label classes<br>
 * <br>
 * Regarding (b) above, this means that the implementations of PathMultiLabelGenerator need to do label to integer index
 * assignment for class labels. Comparatively, PathLabelGenerator can return a Text writable with the label (i.e.,
 * "class_3" or "cat") whereas PathMultiLabelGenerator must return Writables of one of the following types:
 * {@link org.datavec.api.writable.DoubleWritable}, {@link org.datavec.api.writable.FloatWritable},
 * {@link org.datavec.api.writable.IntWritable}, {@link org.datavec.api.writable.LongWritable} or
 * {@link org.datavec.api.writable.NDArrayWritable}
 *
 * @author Alex Black
 */
public interface PathMultiLabelGenerator extends Serializable {

    /**
     * @param uriPath The file or URI path to get the label for
     * @return A list of labels for the specified URI/path
     */
    List<Writable> getLabels(String uriPath);

}
