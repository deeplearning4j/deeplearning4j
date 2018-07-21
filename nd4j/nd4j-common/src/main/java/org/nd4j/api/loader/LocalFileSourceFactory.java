package org.nd4j.api.loader;

public class LocalFileSourceFactory implements SourceFactory {
    @Override
    public Source getSource(String path) {
        return new LocalFileSource(path);
    }
}
