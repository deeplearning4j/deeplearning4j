package org.nd4j.parameterserver.updater.storage;

import org.agrona.concurrent.UnsafeBuffer;
import org.nd4j.aeron.ipc.NDArrayMessage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.rocksdb.Options;
import org.rocksdb.RocksDB;
import org.rocksdb.RocksDBException;
import org.rocksdb.RocksIterator;

import java.nio.ByteBuffer;

/**
 * Created by agibsonccc on 12/2/16.
 */
public class RocksDbStorage extends BaseUpdateStorage implements AutoCloseable {
    static {
        // a static method that loads the RocksDB C++ library.
        RocksDB.loadLibrary();
    }


    private RocksDB db;
    private int size = 0;

    public RocksDbStorage(String dbPath) {
        // that determines the behavior of a database.
        Options options = new Options().setCreateIfMissing(true);
        try {
            // a factory method that returns a RocksDB instance
            db = RocksDB.open(options, dbPath);
            // do something
        } catch (RocksDBException e) {
            // do some error handling

        }
    }

    /**
     * Add an ndarray to the storage
     *
     * @param array the array to add
     */
    @Override
    public void addUpdate(NDArrayMessage array) {
        UnsafeBuffer directBuffer = (UnsafeBuffer) NDArrayMessage.toBuffer(array);
        byte[] data = directBuffer.byteArray();
        if (data == null) {
            data = new byte[directBuffer.capacity()];
            directBuffer.getBytes(0, data, 0, data.length);
        }
        byte[] key = ByteBuffer.allocate(4).putInt(size).array();
        try {
            db.put(key, data);
        } catch (RocksDBException e) {
            throw new RuntimeException(e);
        }

        size++;

    }

    /**
     * The number of updates added
     * to the update storage
     *
     * @return
     */
    @Override
    public int numUpdates() {
        return size;
    }

    /**
     * Clear the array storage
     */
    @Override
    public void clear() {
        RocksIterator iterator = db.newIterator();
        while (iterator.isValid())
            try {
                db.remove(iterator.key());
            } catch (RocksDBException e) {
                throw new RuntimeException(e);
            }
        iterator.close();
        size = 0;
    }

    /**
     * A method for actually performing the implementation
     * of retrieving the ndarray
     *
     * @param index the index of the {@link INDArray} to get
     * @return the ndarray at the specified index
     */
    @Override
    public NDArrayMessage doGetUpdate(int index) {
        byte[] key = ByteBuffer.allocate(4).putInt(index).array();
        try {
            UnsafeBuffer unsafeBuffer = new UnsafeBuffer(db.get(key));
            return NDArrayMessage.fromBuffer(unsafeBuffer, 0);
        } catch (RocksDBException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Close the database
     */
    @Override
    public void close() {
        db.close();
    }
}
