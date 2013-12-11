package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda;

//import net.vivin.digit.DigitImage;    
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.io.FileUtils;

/**
 * Original work: 
 *         http://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
 * 
 * @author josh
 *
 */
public class MNIST_DatasetUtils {




	private String labelFileName;
	private String imageFileName;

	/** the following constants are defined as per the values described at http://yann.lecun.com/exdb/mnist/ **/

	private static final int MAGIC_OFFSET = 0;
	private static final int OFFSET_SIZE = 4; //in bytes

	private static final int LABEL_MAGIC = 2049;
	private static final int IMAGE_MAGIC = 2051;

	private static final int NUMBER_ITEMS_OFFSET = 4;
	private static final int ITEMS_SIZE = 4;

	private static final int NUMBER_OF_ROWS_OFFSET = 8;
	private static final int ROWS_SIZE = 4;
	public static final int ROWS = 28;

	private static final int NUMBER_OF_COLUMNS_OFFSET = 12;
	private static final int COLUMNS_SIZE = 4;
	public static final int COLUMNS = 28;

	private static final int IMAGE_OFFSET = 16;
	private static final int IMAGE_SIZE = ROWS * COLUMNS;

	private static final String trainingFilesURL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
	private static final String trainingFilesFilename = "train-images-idx3-ubyte.gz";
	private static final String trainingFilesFilename_unzipped = "train-images-idx3-ubyte";

	private static final String trainingFileLabelsURL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
	private static final String trainingFileLabelsFilename = "train-labels-idx1-ubyte.gz";
	private static final String trainingFileLabelsFilename_unzipped = "train-labels-idx1-ubyte";

	private static final String metronomeFormattedMNISTFile = "/tmp/MNIST/train-images-MNIST-converted.mne";

	private static File fileDir;
	private static final String LOCAL_DIR_NAME = "MNIST";
	//private static final String TWENTY_NEWS_GROUP_TAR_URL = "http://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydate.tar.gz";
	//private static final String TWENTY_NEWS_GROUP_TAR_FILE_NAME = "20news-bydate.tar.gz";
	/*
	    public static String get20NewsgroupsLocalDataLocation() {

	      File tmpDir = new File("/tmp");
	      if(!tmpDir.isDirectory()) {
	        tmpDir = new File(System.getProperty("java.io.tmpdir"));
	      }
	      File baseDir = new File(tmpDir, LOCAL_DIR_NAME);


	      return baseDir.toString();

	    } 
	 */   


	public static File downloadAndUntar() throws IOException {
		if(fileDir != null) {
			return fileDir;
		}
		// mac gives unique tmp each run and we want to store this persist
		// this data across restarts
		File tmpDir = new File("/tmp");
		if(!tmpDir.isDirectory()) {
			tmpDir = new File(System.getProperty("java.io.tmpdir"));
		}
		File baseDir = new File(tmpDir, LOCAL_DIR_NAME);
		if(!(baseDir.isDirectory() || baseDir.mkdir())) {
			throw new IOException("Could not mkdir " + baseDir);
		}



		// get training records
		File tarFile = new File(baseDir, trainingFilesFilename);

		if(!tarFile.isFile()) {
			FileUtils.copyURLToFile(new URL(trainingFilesURL), tarFile);      
		}

		gunzipFile(baseDir, tarFile);

		// get training records labels - trainingFileLabelsURL
		File tarLabelsFile = new File(baseDir, trainingFileLabelsFilename);

		if(!tarLabelsFile.isFile()) {
			FileUtils.copyURLToFile(new URL(trainingFileLabelsURL), tarLabelsFile);      
		}

		gunzipFile(baseDir, tarLabelsFile);





		fileDir = baseDir;
		return fileDir;
	}

	public static void untarFile(File baseDir, File tarFile) throws IOException {

		System.out.println("Untaring File: " + tarFile.toString());

		Process p = Runtime.getRuntime().exec(String.format("tar -C %s -xvf %s", 
				baseDir.getAbsolutePath(), tarFile.getAbsolutePath()));
		BufferedReader stdError = new BufferedReader(new 
				InputStreamReader(p.getErrorStream()));
		System.out.println("Here is the standard error of the command (if any):\n");
		String s;
		while ((s = stdError.readLine()) != null) {
			System.out.println(s);
		}
		stdError.close();


	}

	public static void gunzipFile(File baseDir, File gzFile) throws IOException {

		System.out.println("gunzip'ing File: " + gzFile.toString());

		Process p = Runtime.getRuntime().exec(String.format("gunzip %s", 
				gzFile.getAbsolutePath()));
		BufferedReader stdError = new BufferedReader(new 
				InputStreamReader(p.getErrorStream()));
		System.out.println("Here is the standard error of the command (if any):\n");
		String s;
		while ((s = stdError.readLine()) != null) {
			System.out.println(s);
		}
		stdError.close();


	}


	public MNIST_DatasetUtils(String labelFileName, String imageFileName) {
		this.labelFileName = labelFileName;
		this.imageFileName = imageFileName;
	}


	/**
	 * 
	 * 784 pixels
	 * 
	 * @param line
	 * @param normalize_base
	 * @return
	 */
	public static String formatMNISTHandwritingColumnToMetronomeRecord(byte[] imageData, int classID, int normalize_base) {

		//String[] parts = line.split(",");

		String inputs = "";

		int max = 0;

		//System.out.println("len: " + parts.length);
		//return "";

		if (imageData.length != 784) {
			System.out.println("Invalid image data!!!!!");
		}


		for ( int x = 0; x < 784; x++ ) {
			//inputs += x + ":" + parts[x].trim() + " ";
			//int tmp = Integer.parseInt(parts[x]);
			int tmp = imageData[x] & 0xFF;
			/*                        if (tmp_max > max) {
	                                max = tmp_max;
	                        }
			 */
			double normalized_val = (double)tmp / (double)normalize_base;
			if (normalized_val > 0.0) {
				inputs += x + ":" + normalized_val + " ";
			}
		}

		//System.out.println("max: " + max);

		//String class_id = parts[64]; // last one
		int cls_id = classID;


		String outputs = "";
		outputs += cls_id + ":1.0";
		//	                outputs += "1:" + parts[5].trim() + " ";
		//	                outputs += "2:" + parts[6].trim() + "";

		return inputs + "| " + outputs + "\n";


	}    

	//	    public List<DigitImage> loadDigitImages() throws IOException {
	public void scanIDXFiles() throws IOException {
		//	        List<DigitImage> images = new ArrayList<DigitImage>();

		int max = 0;

		System.out.println("> Scanning MNIST Files....");
		System.out.println("> Labels: " + labelFileName);
		System.out.println("> Images: " + imageFileName);


		ByteArrayOutputStream labelBuffer = new ByteArrayOutputStream();
		ByteArrayOutputStream imageBuffer = new ByteArrayOutputStream();

		InputStream labelInputStream = new FileInputStream(labelFileName); //this.getClass().getResourceAsStream(labelFileName);
		InputStream imageInputStream = new FileInputStream(imageFileName); //this.getClass().getResourceAsStream(imageFileName);

		int read;
		byte[] buffer = new byte[16384];

		while((read = labelInputStream.read(buffer, 0, buffer.length)) != -1) {
			labelBuffer.write(buffer, 0, read);
		}

		labelBuffer.flush();

		while((read = imageInputStream.read(buffer, 0, buffer.length)) != -1) {
			imageBuffer.write(buffer, 0, read);
		}

		imageBuffer.flush();

		byte[] labelBytes = labelBuffer.toByteArray();
		byte[] imageBytes = imageBuffer.toByteArray();

		byte[] labelMagic = Arrays.copyOfRange(labelBytes, 0, OFFSET_SIZE);
		byte[] imageMagic = Arrays.copyOfRange(imageBytes, 0, OFFSET_SIZE);

		if(ByteBuffer.wrap(labelMagic).getInt() != LABEL_MAGIC)  {
			throw new IOException("Bad magic number in label file!");
		}

		if(ByteBuffer.wrap(imageMagic).getInt() != IMAGE_MAGIC) {
			throw new IOException("Bad magic number in image file!");
		}

		int numberOfLabels = ByteBuffer.wrap(Arrays.copyOfRange(labelBytes, NUMBER_ITEMS_OFFSET, NUMBER_ITEMS_OFFSET + ITEMS_SIZE)).getInt();
		int numberOfImages = ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, NUMBER_ITEMS_OFFSET, NUMBER_ITEMS_OFFSET + ITEMS_SIZE)).getInt();

		if(numberOfImages != numberOfLabels) {
			throw new IOException("The number of labels and images do not match!");
		}

		int numRows = ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, NUMBER_OF_ROWS_OFFSET, NUMBER_OF_ROWS_OFFSET + ROWS_SIZE)).getInt();
		int numCols = ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, NUMBER_OF_COLUMNS_OFFSET, NUMBER_OF_COLUMNS_OFFSET + COLUMNS_SIZE)).getInt();

		if(numRows != ROWS && numRows != COLUMNS) {
			throw new IOException("Bad image. Rows and columns do not equal " + ROWS + "x" + COLUMNS);
		}

		BufferedWriter bw = new BufferedWriter(new FileWriter(metronomeFormattedMNISTFile));


		for(int i = 0; i < numberOfLabels; i++) {
			//	        for(int i = 0; i < 1; i++) {
			int label = labelBytes[OFFSET_SIZE + ITEMS_SIZE + i];
			byte[] imageData = Arrays.copyOfRange(imageBytes, (i * IMAGE_SIZE) + IMAGE_OFFSET, (i * IMAGE_SIZE) + IMAGE_OFFSET + IMAGE_SIZE);

			/*
	            for (int image_index = 0; image_index < imageData.length; image_index++) {
	                    int i2 = imageData[image_index] & 0xFF;
	                    System.out.println(i2);
	            }

			 */
			System.out.print(".");
			String formatted_line = formatMNISTHandwritingColumnToMetronomeRecord(imageData, label, 255);
			bw.write(formatted_line);


			//    images.add(new DigitImage(label, imageData));
			//System.out.println("label: " + label);
		}

		bw.close();

		// return images;
	}

	/*
	        public static void convertIrisNormalizedToMetronome(String filename, String file_out) throws IOException {

	                // src/test/resources/data/iris/iris_data_normalised.txt

	                BufferedWriter bw = new BufferedWriter(new FileWriter(file_out));

	                BufferedReader br = new BufferedReader(new FileReader(filename));
	                String line;
	                while ((line = br.readLine()) != null) {
	                   // process the line.

	                        String formatted_line = formatNISTHandwritingColumnToMetronomeRecord(line, 16);

	                //        System.out.println("old > " + line);
	                //        System.out.println("new > " + formatted_line);
	                        bw.write(formatted_line);

	                }
	                br.close();                
	                bw.close();

	                System.out.println("done...");

	        }        
	 * 
	 */


	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub

		//

		downloadAndUntar();

		MNIST_DatasetUtils util = new MNIST_DatasetUtils( "/tmp/" + LOCAL_DIR_NAME + "/" + trainingFileLabelsFilename_unzipped, "/tmp/" + LOCAL_DIR_NAME + "/" + trainingFilesFilename_unzipped );
		util.scanIDXFiles();


	}

}

