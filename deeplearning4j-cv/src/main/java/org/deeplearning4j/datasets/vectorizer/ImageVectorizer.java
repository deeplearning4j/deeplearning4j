/*
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

package org.deeplearning4j.datasets.vectorizer;

import org.canova.image.loader.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.util.FeatureUtil;

import java.io.File;

/**
 * An image vectorizer takes an input image (RGB) and
 * transforms it in to a data applyTransformToDestination
 * @author Adam Gibson
 *
 */
public class ImageVectorizer implements Vectorizer {

	private File image;
	private ImageLoader loader = new ImageLoader();
	private boolean binarize;
	private boolean normalize;
	private int label;
	private int numLabels;

	/**
	 * Baseline knowledge needed for the vectorizer
	 * @param image the input image to convert
	 * @param numLabels the number of labels
	 * @param label the label of this image
	 */
	public ImageVectorizer(File image, int numLabels, int label) {
		super();
		this.image = image;
		this.numLabels = numLabels;
		this.label = label;
	}


	/**
	 * Binarize the data based on the threshold (anything < threshold is zero)
	 * This  is used for making the image brightness agnostic.
	 * @return builder pattern
	 */
	public ImageVectorizer binarize(int threshold) {
		this.binarize = true;
		this.normalize = false;
		return this;
	}
	
	/**
	 * Binarize the data based on the threshold (anything < threshold is zero)
	 * This  is used for making the image brightness agnostic.
	 * Equivalent to calling (binarze(30))
	 * @return builder pattern
	 */
	public ImageVectorizer binarize() {
		return binarize(30);
	}

	/**
	 * Normalize the input image by row sums
	 * @return builder pattern
	 */
	public ImageVectorizer normalize() {
		this.binarize = false;
		this.normalize = true;
		return this;
	}


	@Override
	public DataSet vectorize() {
		try {
			INDArray d = loader.asMatrix(image);
			INDArray label2 = FeatureUtil.toOutcomeVector(label, numLabels);
			if(normalize) {
				d = d.div(255);
			}
			else if(binarize) {
				for(int i = 0; i < d.length(); i++) {
					double curr = (double) d.getScalar(i).element();
					int threshold = 30;
					if(curr > threshold) {
						d.putScalar(i, 1);
					}
					else 
						d.putScalar(i, 0);


				}
			}


			return new DataSet(d,label2);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}


}
