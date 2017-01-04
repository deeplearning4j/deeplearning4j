package org.deeplearning4j.keras;

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import org.apache.commons.io.FileUtils;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.reflections.Reflections;
import org.reflections.scanners.ResourcesScanner;

import javax.annotation.Nullable;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Set;
import java.util.regex.Pattern;

import static org.deeplearning4j.keras.StringsEndsWithPredicate.endsWith;

public class DeepLearning4jEntryPointTest {

    private DeepLearning4jEntryPoint deepLearning4jEntryPoint = new DeepLearning4jEntryPoint();

    public
    @Rule
    ExpectedException thrown = ExpectedException.none();

    @Test
    public void shouldFitTheSampleSequentialModel() throws Exception {
        // Given
        final File model = prepareResource("theano_mnist/model.h5");
        final Path features = prepareFeatures("theano_mnist");
        final Path labels = prepareLabels("theano_mnist");

        EntryPointFitParameters entryPointParameters = EntryPointFitParameters
                .builder()
                .modelFilePath(model.getAbsolutePath())
                .trainFeaturesDirectory(features.toAbsolutePath().toString())
                .trainLabelsDirectory(labels.toAbsolutePath().toString())
                .batchSize(128)
                .nbEpoch(2)
                .type(KerasModelType.SEQUENTIAL)
                .build();

        // When
        deepLearning4jEntryPoint.fit(entryPointParameters);

        // Then
        // fall through - the rule will fail the test execution if an exception is thrown
    }

    private Path prepareLabels(String dataSet) throws IOException {
        return prepareDataSet(dataSet, "labels");
    }

    private Path prepareFeatures(String dataSet) throws IOException {
        return prepareDataSet(dataSet, "features");
    }

    private Path prepareDataSet(String dataSet, String batchesSubset) throws IOException {
        Set<String> batchFiles = listBatchFiles(dataSet + "/" + batchesSubset);
        final Path tempDirectory = Files.createTempDirectory("dl4j-" + batchesSubset);

        for (String batchFile : batchFiles) {
            String batchFileName = Paths.get(batchFile).getFileName().toString();
            copyClasspathResourceToFile(batchFile, targetFile(tempDirectory, batchFileName));
        }
        return tempDirectory;
    }

    private File targetFile(Path tempFeaturesDirectory, String batchFileName) {
        return Paths.get(
                tempFeaturesDirectory.toFile().getAbsolutePath().toString(),
                batchFileName
        ).toFile();
    }

    private Set<String> listBatchFiles(String classpathDirectory) {
        return new Reflections(classpathDirectory, new ResourcesScanner()).getResources(endsWith(".h5"));
    }

    private File prepareResource(String resourceName) throws IOException {
        File file = File.createTempFile("dl4j", "test");
        file.delete();
        file.deleteOnExit();
        copyClasspathResourceToFile(resourceName, file);
        return file;
    }

    private void copyClasspathResourceToFile(String resourceName, File file) throws IOException {
        FileUtils.copyInputStreamToFile(this.getClass().getClassLoader().getResourceAsStream(resourceName), file);
    }

}