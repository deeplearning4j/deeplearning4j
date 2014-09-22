package org.deeplearning4j.tsne;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.plot.Tsne;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Created by agibsonccc on 9/20/14.
 */
public class TsneExample {

    public static void main(String[] args) throws Exception  {
        Tsne tsne = new Tsne();
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        fetcher.fetch(10);
        DataSet d2 = fetcher.next();

        tsne.plot(d2.getFeatureMatrix(),2,0.2f,28);
    }



}
