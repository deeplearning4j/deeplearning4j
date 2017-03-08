package org.deeplearning4j.text.documentiterator;

import lombok.NonNull;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This is simple filesystem-based LabelAware iterator.
 * It assumes that you have one or more folders organized in the following way:
 * 1st level subfolder: label name
 * 2nd level: bunch of documents for that label
 *
 * You can have as many label folders as you want, as well.
 *
 * @author raver119@gmail.com
 */
public class FileLabelAwareIterator implements LabelAwareIterator {
    protected List<File> files;
    protected AtomicInteger position = new AtomicInteger(0);
    protected LabelsSource labelsSource;

    /*
        Please keep this method protected, it's used in tests
     */
    protected FileLabelAwareIterator() {

    }

    protected FileLabelAwareIterator(@NonNull List<File> files, @NonNull LabelsSource source) {
        this.files = files;
        this.labelsSource = source;
    }

    @Override
    public boolean hasNextDocument() {
        return position.get() < files.size();
    }


    @Override
    public LabelledDocument nextDocument() {
        File fileToRead = files.get(position.getAndIncrement());
        String label = fileToRead.getParentFile().getName();
        try {
            LabelledDocument document = new LabelledDocument();
            BufferedReader reader = new BufferedReader(new FileReader(fileToRead));
            StringBuilder builder = new StringBuilder();
            String line = "";
            while ((line = reader.readLine()) != null)
                builder.append(line).append(" ");

            reader.close();

            document.setContent(builder.toString());
            document.addLabel(label);

            try {
                reader.close();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            return document;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public boolean hasNext() {
        return hasNextDocument();
    }

    @Override
    public LabelledDocument next() {
        return nextDocument();
    }

    @Override
    public void remove() {
        // no-op
    }

    @Override
    public void shutdown() {
        // no-op
    }

    @Override
    public void reset() {
        position.set(0);
    }

    @Override
    public LabelsSource getLabelsSource() {
        return labelsSource;
    }

    public static class Builder {
        protected List<File> foldersToScan = new ArrayList<>();

        public Builder() {

        }

        /**
         * Root folder for labels -> documents.
         * Each subfolder name will be presented as label, and contents of this folder will be represented as LabelledDocument, with label attached
         *
         * @param folder folder to be scanned for labels and files
         * @return
         */
        public Builder addSourceFolder(@NonNull File folder) {
            foldersToScan.add(folder);
            return this;
        }

        public FileLabelAwareIterator build() {
            // search for all files in all folders provided
            List<File> fileList = new ArrayList<>();
            List<String> labels = new ArrayList<>();

            for (File file : foldersToScan) {
                if (!file.isDirectory())
                    continue;



                File[] files = file.listFiles();
                if (files == null || files.length == 0)
                    continue;


                for (File fileLabel : files) {
                    if (!fileLabel.isDirectory())
                        continue;

                    if (!labels.contains(fileLabel.getName()))
                        labels.add(fileLabel.getName());

                    File[] docs = fileLabel.listFiles();
                    if (docs == null || docs.length == 0)
                        continue;

                    for (File fileDoc : docs) {
                        if (!fileDoc.isDirectory())
                            fileList.add(fileDoc);
                    }
                }
            }
            LabelsSource source = new LabelsSource(labels);
            FileLabelAwareIterator iterator = new FileLabelAwareIterator(fileList, source);

            return iterator;
        }
    }
}
