package org.nd4j.linalg.dataset;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Auto balance mini batches by label.
 * @author Adam Gibson
 */
@AllArgsConstructor
@Builder
@Data
public class BalanceMinibatches {
    private DataSetIterator dataSetIterator;
    private int numLabels;
    private Map<Integer, List<File>> paths = Maps.newHashMap();
    private int miniBatchSize = -1;
    private File rootDir = new File("minibatches");
    private File rootSaveDir = new File("minibatchessave");
    private List<File> labelRootDirs = new ArrayList<>();
    private DataNormalization dataNormalization;

    /**
     * Generate a balanced
     * dataset minibatch fileset.
     */
    public void balance() {
        if (!rootDir.exists())
            rootDir.mkdirs();
        if (!rootSaveDir.exists())
            rootSaveDir.mkdirs();

        if (paths == null)
            paths = Maps.newHashMap();
        if (labelRootDirs == null)
            labelRootDirs = Lists.newArrayList();

        for (int i = 0; i < numLabels; i++) {
            paths.put(i, new ArrayList<File>());
            labelRootDirs.add(new File(rootDir, String.valueOf(i)));
        }


        //lay out each example in their respective label directories tracking the paths along the way
        while (dataSetIterator.hasNext()) {
            DataSet next = dataSetIterator.next();
            //infer minibatch size from iterator
            if (miniBatchSize < 0)
                miniBatchSize = next.numExamples();
            for (int i = 0; i < next.numExamples(); i++) {
                DataSet currExample = next.get(i);
                if (!labelRootDirs.get(currExample.outcome()).exists())
                    labelRootDirs.get(currExample.outcome()).mkdirs();

                //individual example will be saved to: labelrootdir/examples.size()
                File example = new File(labelRootDirs.get(currExample.outcome()),
                                String.valueOf(paths.get(currExample.outcome()).size()));
                currExample.save(example);
                paths.get(currExample.outcome()).add(example);
            }
        }

        int numsSaved = 0;
        //loop till all file paths have been removed
        while (!paths.isEmpty()) {
            List<DataSet> miniBatch = new ArrayList<>();
            while (miniBatch.size() < miniBatchSize && !paths.isEmpty()) {
                for (int i = 0; i < numLabels; i++) {
                    if (paths.get(i) != null && !paths.get(i).isEmpty()) {
                        DataSet d = new DataSet();
                        d.load(paths.get(i).remove(0));
                        miniBatch.add(d);
                    } else
                        paths.remove(i);
                }
            }

            if (!rootSaveDir.exists())
                rootSaveDir.mkdirs();
            //save with an incremental count of the number of minibatches saved
            if (!miniBatch.isEmpty()) {
                DataSet merge = DataSet.merge(miniBatch);
                if (dataNormalization != null)
                    dataNormalization.transform(merge);
                merge.save(new File(rootSaveDir, String.format("dataset-%d.bin", numsSaved++)));
            }


        }

    }

}
