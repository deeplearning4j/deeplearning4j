package org.deeplearning4j.text.documentiterator.interoperability;

import lombok.NonNull;
import org.deeplearning4j.text.documentiterator.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

/**
 * Simple class providing compatibility between DocumentIterator/LabelAwareDocumentIterator and LabelAwareIterator
 *
 * @author raver119@gmail.com
 */
public class DocumentIteratorConverter implements LabelAwareIterator {

    protected DocumentIterator backendIterator;
    protected LabelsSource generator;

    public DocumentIteratorConverter(@NonNull LabelAwareDocumentIterator iterator) {
        this.backendIterator = iterator;
    }

    public DocumentIteratorConverter(@NonNull DocumentIterator iterator, @NonNull LabelsSource generator) {
        this.backendIterator = iterator;
        this.generator = generator;
    }

    @Override
    public boolean hasNextDocument() {
        return backendIterator.hasNext();
    }

    @Override
    public LabelledDocument nextDocument() {
        try {
            LabelledDocument document = new LabelledDocument();

            document.setContent(readStream(backendIterator.nextDocument()));

            if (backendIterator instanceof LabelAwareDocumentIterator) {
                String currentLabel = ((LabelAwareDocumentIterator) backendIterator).currentLabel();
                document.setLabel(currentLabel);
                generator.storeLabel(currentLabel);
            } else document.setLabel(generator.nextLabel());

            return document;
        } catch (Exception e) {
            // we just publish caught exception, no magic or automation here
            throw new RuntimeException(e);
        }
    }

    @Override
    public void reset() {
        generator.reset();
        backendIterator.reset();
    }

    @Override
    public LabelsSource getLabelsSource() {
        return generator;
    }

    protected String readStream(InputStream stream) throws IOException {
        StringBuilder builder = new StringBuilder();

        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        String line = "";
        while ((line = reader.readLine()) != null) {
            builder.append(line);
        }
        return builder.toString();
    }
}
