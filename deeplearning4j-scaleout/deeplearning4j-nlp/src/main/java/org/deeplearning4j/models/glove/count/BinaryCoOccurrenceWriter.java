package org.deeplearning4j.models.glove.count;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;

/**
 * @author raver119@gmail.com
 */
public class BinaryCoOccurrenceWriter<T extends SequenceElement> implements CoOccurrenceWriter<T> {
    private File file;
    private DataOutputStream outputStream;

    public BinaryCoOccurrenceWriter(@NonNull File file) {
        this.file = file;

        try {
            outputStream = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void writeObject(@NonNull CoOccurrenceWeight<T> object) {
        try {
            outputStream.writeInt(object.getElement1().getIndex());
            outputStream.writeInt(object.getElement2().getIndex());
            outputStream.writeDouble(object.getWeight());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void finish() {
        try {
            outputStream.flush();
        } catch (Exception e) {
            ;
        }

        try {
            outputStream.close();
        } catch (Exception e) {
            ;
        }
    }
}
