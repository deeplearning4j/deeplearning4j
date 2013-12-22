package com.ccc.sendalyzeit.textanalytics.algorithms.datasets;


import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * 
 * MNIST database label file.
 * 
 */
public class MnistLabelFile extends MnistDbFile {

    /**
     * Creates new MNIST database label file ready for reading.
     * 
     * @param name
     *            the system-dependent filename
     * @param mode
     *            the access mode
     * @throws IOException
     * @throws FileNotFoundException
     */
    public MnistLabelFile(String name, String mode) throws FileNotFoundException, IOException {
        super(name, mode);
    }

    /**
     * Reads the integer at the current position.
     * 
     * @return integer representing the label
     * @throws IOException
     */
    public int readLabel() throws IOException {
        return readUnsignedByte();
    }

    @Override
    protected int getMagicNumber() {
        return 2049;
    }
}
