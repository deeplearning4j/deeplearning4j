package org.deeplearning4j.tsne;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.plot.Tsne;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 9/20/14.
 */
public class TsneExample {

    public static void main(String[] args) throws Exception  {
        Tsne tsne = new Tsne.Builder().setMaxIter(10000)
                .learningRate(1e-3f).useAdaGrad(true)
                .normalize(false).usePca(true).build();
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        fetcher.fetch(10);
        DataSet d2 = fetcher.next();
        List<String> list = new ArrayList<>();
        for(int i = 0; i < d2.numExamples(); i++)
            list.add(String.valueOf(d2.get(i).outcome()));
        tsne.plot(d2.getFeatureMatrix(),2,28,list);
    }



}
