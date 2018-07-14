package org.deeplearning4j.spark.data.loader;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.nd4j.api.loader.Source;
import org.nd4j.api.loader.SourceFactory;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

@AllArgsConstructor
@NoArgsConstructor
public class RemoteFileSourceFactory implements SourceFactory {
    private FileSystem fileSystem;

    @Override
    public Source getSource(String path) {
        if(fileSystem == null){
            try {
                fileSystem = FileSystem.get(new URI(path), new Configuration());
            } catch (IOException | URISyntaxException u){
                throw new RuntimeException(u);
            }
        }

        return new RemoteFileSource(path, fileSystem);
    }
}
