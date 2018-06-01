package org.deeplearning4j.zoo.util;

import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.resources.Downloader;

import java.io.*;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Base functionality for helper classes that return label descriptions.
 *
 * @author saudet
 */
public abstract class BaseLabels implements Labels {

    protected ArrayList<String> labels;

    /** Override {@link #getLabels()} when using this constructor. */
    protected BaseLabels() throws IOException {
        this.labels = getLabels();
    }

    /**
     * No need to override anything with this constructor.
     *
     * @param textResource name of a resource containing labels as a list in a text file.
     * @throws IOException 
     */
    protected BaseLabels(String textResource) throws IOException {
        this.labels = getLabels(textResource);
    }

    /**
     * Override to return labels when not calling {@link #BaseLabels(String)}.
     */
    protected ArrayList<String> getLabels() throws IOException {
        return null;
    }

    /**
     * Returns labels based on the text file resource.
     */
    protected ArrayList<String> getLabels(String textResource) throws IOException {
        ArrayList<String> labels = new ArrayList<>();
        File resourceFile = getResourceFile();  //Download if required
        try (InputStream is = new BufferedInputStream(new FileInputStream(resourceFile)); Scanner s = new Scanner(is)) {
            while (s.hasNextLine()) {
                labels.add(s.nextLine());
            }
        }
        return labels;
    }

    @Override
    public String getLabel(int n) {
        Preconditions.checkArgument(n >= 0 && n < labels.size(), "Invalid index: %s. Must be in range" +
                "0 <= n < %s", n, labels.size());
        return labels.get(n);
    }

    @Override
    public List<List<ClassPrediction>> decodePredictions(INDArray predictions, int n) {
        Preconditions.checkState(predictions.size(1) == labels.size(), "Invalid input array:" +
                " expected array with size(1) equal to numLabels (%s), got array with shape %s", labels.size(), predictions.shape());

        // FIXME: int cast
        int rows = (int) predictions.size(0);
        int cols = (int) predictions.size(1);
        if (predictions.isColumnVectorOrScalar()) {
            predictions = predictions.ravel();
            rows = (int) predictions.size(0);
            cols = (int) predictions.size(1);
        }
        List<List<ClassPrediction>> descriptions = new ArrayList<>();
        for (int batch = 0; batch < rows; batch++) {
            INDArray result = predictions.getRow(batch);
            result = Nd4j.vstack(Nd4j.linspace(0, cols, cols), result);
            result = Nd4j.sortColumns(result, 1, false);
            List<ClassPrediction> current = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                int label = result.getInt(0, i);
                double prob = result.getDouble(1, i);
                current.add(new ClassPrediction(label, getLabel(label), prob));
            }
            descriptions.add(current);
        }
        return descriptions;
    }

    /**
     * @return URL of the resource to download
     */
    protected abstract URL getURL();

    /**
     * @return Name of the resource (used for inferring local storage parent directory)
     */
    protected abstract String resourceName();

    /**
     * @return MD5 of the resource at getURL()
     */
    protected abstract String resourceMD5();

    /**
     * Download the resource at getURL() to the local resource directory, and return the local copy as a File
     *
     * @return File of the local resource
     */
    protected File getResourceFile(){

        URL url = getURL();
        String urlString = url.toString();
        String filename = urlString.substring(urlString.lastIndexOf('/')+1);
        File resourceDir = DL4JResources.getDirectory(ResourceType.RESOURCE, resourceName());
        File localFile = new File(resourceDir, filename);

        String expMD5 = resourceMD5();
        if(localFile.exists()){
            try{
                if(Downloader.checkMD5OfFile(expMD5, localFile)){
                    return localFile;
                }
            } catch (IOException e){
                //Ignore
            }
            //MD5 failed
            localFile.delete();
        }

        //Download
        try {
            Downloader.download(resourceName(), url, localFile, expMD5, 3);
        } catch (IOException e){
            throw new RuntimeException("Error downloading labels",e);
        }

        return localFile;
    }

}
