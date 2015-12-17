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

package org.deeplearning4j.datasets.fetchers;

import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.base.LFWFetcher;
import org.deeplearning4j.util.MathUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.FeatureUtil;

import javax.imageio.ImageIO;
import javax.imageio.stream.ImageOutputStream;
import javax.imageio.stream.MemoryCacheImageOutputStream;


/**
 * Data fetcher for the LFW faces dataset
 * @author Adam Gibson
 *
 */
public class LFWDataFetcher extends BaseDataFetcher {

	public final static int NUM_IMAGES = 13233;
	public final static int NUM_LABELS = 1680;
	protected final static String BASE_DIR = System.getProperty("user.home");
	protected static String fileDir = FilenameUtils.concat(BASE_DIR, "LFW");
	protected static String subsetFileDir = FilenameUtils.concat(BASE_DIR, "lfw-a/lfw");
	protected static boolean isSubset = false;
	protected static final String[] allowedFormats = {"jpg", "jpeg", "JPG", "JPEG"};

	protected boolean binarize = false;
	protected int width = 250;
	protected int height = 250;
	protected int channels = 3;
	protected LFWFetcher loader;
	protected int[] order;
	protected Random rng;
	protected boolean shuffle;

	protected List<String> imageList = new ArrayList<>();
	protected List<String> outcomeList = new ArrayList<>();
	protected List<String> labels;
	protected Map<String,String> fileNameMap = new LinkedHashMap<>();

	public LFWDataFetcher() {}

	public LFWDataFetcher(boolean isSubset){
		 this.isSubset = isSubset;
	}

	public LFWDataFetcher(int imageWidth, int imageHeight, int channels) {
		this(imageWidth, imageHeight, channels, fileDir, isSubset);
	}

	public LFWDataFetcher(int imageWidth, int imageHeight, int channels, boolean isSubset){
		this(imageWidth, imageHeight, channels, subsetFileDir, isSubset);
	}

	public LFWDataFetcher(int imageWidth, int imageHeight, int channels, String fileDir, boolean isSubset) {
		// TODO if width and height less than 250 then need to resize...
		this.width = imageWidth;
		this.height = imageHeight;
		this.channels = channels;
		this.fileDir = fileDir;
		this.isSubset = isSubset;
		totalExamples = NUM_IMAGES;
		inputColumns = width * height * channels;
		LFWFetcher lFetch = new LFWFetcher(fileDir, isSubset);

		if(!lFetch.lfwExists())
			lFetch.fetch();
		getImgOutLists();


//		try {
//			BufferedImage imagePath = ImageIO.read(new File(FilenameUtils.concat(fileDir, LFWFetcher.lfwData.get("filesFilenameUnzipped"))));
//		} catch (IOException e){
//
//		}
//		String labelPath = FilenameUtils.concat(fileDir, LFWFetcher.lfwLabel.get("filesFilenameUnzipped"));

	}

	@Override
	public void fetch(int numExamples) {
		if(!hasMore())
			throw new IllegalStateException("Unable to getFromOrigin more; there are no more images");


		//we need to ensure that we don't overshoot the number of examples total
		List<DataSet> toConvert = new ArrayList<>(numExamples);
		for(int i = 0; i < numExamples; i++,cursor++) {
			if(!hasMore())
				break;
			toConvert.add(getDataFor(cursor));
		}

		initializeCurrFromList(toConvert);
		setLabelNames(outcomeList); // submitting list of names in order of the data that is loaded
	}

	@Override
	public void reset() {
		cursor = 0;
		curr = null;
		if(shuffle) MathUtils.shuffleArray(order, rng);
	}


	@Override
	public DataSet next() {
		DataSet next = super.next();
		return next;
	}

	protected void defineLabels(){
		labels = new ArrayList<>(new HashSet(fileNameMap.values()));
		numOutcomes = labels.size();
	}

	public DataSet getDataFor(int i) {

		// read image
		try {
			File imgFile = new File(imageList.get(i));
			ByteArrayOutputStream tmpBOS = new ByteArrayOutputStream();
			ImageOutputStream stream = new MemoryCacheImageOutputStream(tmpBOS);
			BufferedImage tmpBImg = ImageIO.read(imgFile);
			ImageIO.write(tmpBImg, "jpg", stream);
			byte[] imageInByte = tmpBOS.toByteArray();
			stream.close();
			tmpBOS.close();

			INDArray in = Nd4j.create(1, imageInByte.length);
			for( int j=0; j < imageInByte.length; j++ ){
				in.putScalar(j, ((int) imageInByte[j]));
//				in.putScalar(j, ((int) imageInByte[j]) & 0xFF);
			}

//			if(binarize) {
//				for(int d = 0; d < in.length(); d++) {
//					if(in.getDouble(d) > 30) {
//						in.putScalar(d,1);
//					}
//					else {
//						in.putScalar(d,0);
//					}
//				}
//			} else {
//				in.divi(255);
//			}

			// create outcome vector that reps the number of outcomes and label position
			return new DataSet(in, FeatureUtil.toOutcomeVector(labels.indexOf(outcomeList.get(i)), numOutcomes));

		} catch (IOException e) {
			log.debug(e.getMessage());
			throw new IllegalStateException("Unable to getFromOrigin data for image " + i + " for path " + imageList.get(i));
		}
	}


	protected void getImgOutLists() {
		String name, label;
		Collection<File> subFiles = FileUtils.listFiles(new File(fileDir), allowedFormats, true);
		for(File file : subFiles) {
			name = FilenameUtils.getBaseName(file.getName());
			label = name.split(".[0-9]+")[0];
			imageList.add(file.toString());
			outcomeList.add(label);
			fileNameMap.put(name, label);
		}
		totalExamples = imageList.size();
		order = new int[totalExamples];
		defineLabels();
	}

	public int getNumNames() {
		return numOutcomes;
	}

	public int getNumPixels() {
		return width * height;
	}

	public String getLabelName(String path) {
		return fileNameMap.get(path);
	}
}
