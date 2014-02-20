package org.deeplearning4j.base;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.util.ArrayUtil;
import org.deeplearning4j.util.ImageLoader;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class LFWLoader {

	private File baseDir = new File(System.getProperty("java.io.tmpdir"));
	public final static String LFW = "lfw";
	private File lfwDir = new File(baseDir,LFW);
	public final static String LFW_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz";
	private File lfwTarFile = new File(lfwDir,"lfw.tgz");
	private static Logger log = LoggerFactory.getLogger(LFWLoader.class);
	private int numNames;
	private int numPixelColumns;
	private ImageLoader loader = new ImageLoader();
	private List<String> images = new ArrayList<String>();
	private List<String> outcomes = new ArrayList<String>();
	
	public void getIfNotExists() throws Exception {
		if(!lfwDir.exists()) {
			lfwDir.mkdir();
			FileUtils.copyURLToFile(new URL(LFW_URL), lfwTarFile);
			//untar to /tmp/lfw
			untarFile(baseDir,lfwTarFile);
			
		}
		
		File firstImage = lfwDir.listFiles()[0].listFiles()[0];
		//number of input neurons
		numPixelColumns = ArrayUtil.flatten(loader.fromFile(firstImage)).length;
		
		//each subdir is a person
		numNames = lfwDir.getAbsoluteFile().listFiles().length;
		
		Collection<File> allImages = FileUtils.listFiles(lfwDir, org.apache.commons.io.filefilter.FileFileFilter.FILE, org.apache.commons.io.filefilter.DirectoryFileFilter.DIRECTORY);
		for(File f : allImages) {
			images.add(f.getAbsolutePath());
		}
		for(File dir : lfwDir.getAbsoluteFile().listFiles())
			outcomes.add(dir.getAbsolutePath());
		
	}
	
	
	
	public Pair<DoubleMatrix,DoubleMatrix> convertListPairs(List<Pair<DoubleMatrix,DoubleMatrix>> images) {
		DoubleMatrix inputs = new DoubleMatrix(images.size(),numPixelColumns);
		DoubleMatrix outputs = new DoubleMatrix(images.size(),numNames);
		
		for(int i = 0; i < images.size(); i++) {
			inputs.putRow(i,images.get(i).getFirst());
			outputs.putRow(i,images.get(i).getSecond());
		}
		return new Pair<DoubleMatrix,DoubleMatrix>(inputs,outputs);
	}
	
	
	
	public Pair<DoubleMatrix,DoubleMatrix> getDataFor(int i) {
		File image = new File(images.get(i));
		int outcome = outcomes.indexOf(image.getParentFile().getAbsolutePath());
		try {
			return new Pair<DoubleMatrix,DoubleMatrix>(loader.asRowVector(image),MatrixUtil.toOutcomeVector(outcome, outcomes.size()));
		} catch (Exception e) {
			throw new IllegalStateException("Unable to get data for image " + i + " for path " + images.get(i));
		}
	}
	
	/**
	 * Get the first num found images
	 * @param num the number of images to get
	 * @return 
	 * @throws Exception
	 */
	public List<Pair<DoubleMatrix,DoubleMatrix>> getFirst(int num) throws Exception {
		List<Pair<DoubleMatrix,DoubleMatrix>> ret = new ArrayList<>(num);
		File[] files = lfwDir.listFiles();
		int label = 0;
		for(File file : files) {
			ret.addAll(getImages(label,file));
			label++;
			if(ret.size() >= num)
				break;
		}
		
		return ret;
	}
	
	public Pair<DoubleMatrix,DoubleMatrix> getAllImagesAsMatrix() throws Exception {
		List<Pair<DoubleMatrix,DoubleMatrix>> images = getImagesAsList();
		return convertListPairs(images);
	}
	

	public Pair<DoubleMatrix,DoubleMatrix> getAllImagesAsMatrix(int numRows) throws Exception {
		List<Pair<DoubleMatrix,DoubleMatrix>> images = getImagesAsList().subList(0, numRows);
		return convertListPairs(images);
	}
	
	public List<Pair<DoubleMatrix,DoubleMatrix>> getImagesAsList() throws Exception {
		List<Pair<DoubleMatrix,DoubleMatrix>> list = new ArrayList<Pair<DoubleMatrix,DoubleMatrix>>();
		File[] dirs = lfwDir.listFiles();
		for(int i = 0; i < dirs.length; i++) {
			list.addAll(getImages(i,dirs[i]));
		}
		return list;
	}
	
	public List<Pair<DoubleMatrix,DoubleMatrix>> getImages(int label,File file) throws Exception {
		File[] images = file.listFiles();
		List<Pair<DoubleMatrix,DoubleMatrix>> ret = new ArrayList<>();
		for(File f : images)
			ret.add(fromImageFile(label,f));
		return ret;
	}
	
	
	public Pair<DoubleMatrix,DoubleMatrix> fromImageFile(int label,File image) throws Exception {
		DoubleMatrix outcome = MatrixUtil.toOutcomeVector(label, numNames);
		DoubleMatrix image2 = MatrixUtil.toMatrix(loader.flattenedImageFromFile(image));
		return new Pair<>(image2,outcome);
	}
	
	

	public  void untarFile(File baseDir, File tarFile) throws IOException {

		log.info("Untaring File: " + tarFile.toString());

		Process p = Runtime.getRuntime().exec(String.format("tar -C %s -xvf %s", 
				baseDir.getAbsolutePath(), tarFile.getAbsolutePath()));
		BufferedReader stdError = new BufferedReader(new 
				InputStreamReader(p.getErrorStream()));
		log.info("Here is the standard error of the command (if any):\n");
		String s;
		while ((s = stdError.readLine()) != null) {
			log.info(s);
		}
		stdError.close();


	}

	public static void gunzipFile(File baseDir, File gzFile) throws IOException {

		log.info("gunzip'ing File: " + gzFile.toString());

		Process p = Runtime.getRuntime().exec(String.format("gunzip %s", 
				gzFile.getAbsolutePath()));
		BufferedReader stdError = new BufferedReader(new 
				InputStreamReader(p.getErrorStream()));
		log.info("Here is the standard error of the command (if any):\n");
		String s;
		while ((s = stdError.readLine()) != null) {
			log.info(s);
		}
		stdError.close();


	}



	public int getNumNames() {
		return numNames;
	}



	public int getNumPixelColumns() {
		return numPixelColumns;
	}
	
	

}
