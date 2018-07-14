package org.deeplearning4j.spark.data.loader;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.nd4j.api.loader.Source;

import java.io.IOException;
import java.io.InputStream;

@AllArgsConstructor
public class RemoteFileSource implements Source {
    public static final int DEFAULT_BUFFER_SIZE = 4*1024*2014;
    @Getter
    private String path;
    private final FileSystem fileSystem;
    private final int bufferSize;

    public RemoteFileSource(String path, FileSystem fileSystem){
        this(path, fileSystem, DEFAULT_BUFFER_SIZE);
    }

    @Override
    public InputStream getInputStream() throws IOException {
        return fileSystem.open(new Path(path), bufferSize);
    }
}
