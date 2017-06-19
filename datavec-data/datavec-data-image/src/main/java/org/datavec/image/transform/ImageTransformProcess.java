package org.datavec.image.transform;

import com.google.common.collect.Sets;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ClassUtils;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.api.util.reflections.DataVecSubTypesScanner;
import org.datavec.api.writable.Writable;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.PropertyAccessor;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.shade.jackson.databind.introspect.AnnotatedClass;
import org.nd4j.shade.jackson.databind.jsontype.NamedType;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;
import org.nd4j.shade.jackson.datatype.joda.JodaModule;
import org.reflections.ReflectionUtils;
import org.reflections.Reflections;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;
import org.reflections.util.FilterBuilder;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Modifier;
import java.net.URI;
import java.net.URL;
import java.util.*;

/**
 * Created by kepricon on 17. 5. 23.
 */
@Data
@Slf4j
@NoArgsConstructor
public class ImageTransformProcess {

    private List<ImageTransform> transformList;
    private int seed;

    private static Set<Class<?>> subtypesClassCache = null;
    private static ObjectMapper jsonMapper = initMapperJson();
    private static ObjectMapper yamlMapper = initMapperYaml();

    public ImageTransformProcess(int seed, ImageTransform... transforms) {
        this.seed = seed;
        this.transformList = Arrays.asList(transforms);
    }

    public ImageTransformProcess(int seed, List<ImageTransform> transformList) {
        this.seed = seed;
        this.transformList = transformList;
    }

    public ImageTransformProcess(Builder builder) {
        this(builder.seed, builder.transformList);
    }

    public List<Writable> execute(List<Writable> image) {
        throw new UnsupportedOperationException();
    }

    public INDArray executeArray(ImageWritable image) throws IOException {
        Random random = null;
        if (seed != 0) {
            random = new Random(seed);
        }

        ImageWritable currentImage = image;
        for (ImageTransform transform : transformList) {
            currentImage = transform.transform(currentImage, random);
        }

        NativeImageLoader imageLoader = new NativeImageLoader();
        return imageLoader.asMatrix(currentImage);
    }

    public ImageWritable execute(ImageWritable image) throws IOException {
        Random random = null;
        if (seed != 0) {
            random = new Random(seed);
        }

        ImageWritable currentImage = image;
        for (ImageTransform transform : transformList) {
            currentImage = transform.transform(currentImage, random);
        }

        return currentImage;
    }

    public ImageWritable transformFileUriToInput(URI uri) throws IOException {

        NativeImageLoader imageLoader = new NativeImageLoader();
        ImageWritable img = imageLoader.asWritable(new File(uri));

        return img;
    }

    /**
     * Convert the ImageTransformProcess to a JSON string
     *
     * @return ImageTransformProcess, as JSON
     */
    public String toJson() {
        try {
            return jsonMapper.writeValueAsString(this);
        } catch (JsonProcessingException e) {
            //Ignore the first exception, try reinitializing subtypes for custom transforms etc
        }

        jsonMapper = reinitializeMapperWithSubtypes(jsonMapper);

        try {
            return jsonMapper.writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Convert the ImageTransformProcess to a YAML string
     *
     * @return ImageTransformProcess, as YAML
     */
    public String toYaml() {
        try {
            return yamlMapper.writeValueAsString(this);
        } catch (JsonProcessingException e) {
            //Ignore the first exception, try reinitializing subtypes for custom transforms etc
        }

        yamlMapper = reinitializeMapperWithSubtypes(yamlMapper);

        try {
            return yamlMapper.writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Deserialize a JSON String (created by {@link #toJson()}) to a ImageTransformProcess
     *
     * @return ImageTransformProcess, from JSON
     */
    public static ImageTransformProcess fromJson(String json) {
        try {
            return jsonMapper.readValue(json, ImageTransformProcess.class);
        } catch (IOException e) {
            //Ignore the first exception, try reinitializing subtypes for custom transforms etc
        }

        jsonMapper = reinitializeMapperWithSubtypes(jsonMapper);

        try {
            return jsonMapper.readValue(json, ImageTransformProcess.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Deserialize a JSON String (created by {@link #toJson()}) to a ImageTransformProcess
     *
     * @return ImageTransformProcess, from JSON
     */
    public static ImageTransformProcess fromYaml(String yaml) {
        try {
            return yamlMapper.readValue(yaml, ImageTransformProcess.class);
        } catch (IOException e) {
            //Ignore the first exception, try reinitializing subtypes for custom transforms etc
        }

        yamlMapper = reinitializeMapperWithSubtypes(yamlMapper);

        try {
            return yamlMapper.readValue(yaml, ImageTransformProcess.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static ObjectMapper reinitializeMapperWithSubtypes(ObjectMapper mapper) {
        //Register concrete subtypes for JSON serialization

        List<Class<?>> classes =
                Arrays.<Class<?>>asList(ImageTransform.class);
        List<String> classNames = new ArrayList<>(6);
        for (Class<?> c : classes)
            classNames.add(c.getName());

        // First: scan the classpath and find all instances of the 'baseClasses' classes

        if (subtypesClassCache == null) {
            List<Class<?>> interfaces =
                    Arrays.<Class<?>>asList(ImageTransform.class);
            List<Class<?>> classesList = Arrays.<Class<?>>asList();

            Collection<URL> urls = ClasspathHelper.forClassLoader();
            List<URL> scanUrls = new ArrayList<>();
            for (URL u : urls) {
                String path = u.getPath();
                if (!path.matches(".*/jre/lib/.*jar")) { //Skip JRE/JDK JARs
                    scanUrls.add(u);
                }
            }

            Reflections reflections = new Reflections(
                    new ConfigurationBuilder().filterInputsBy(new FilterBuilder().exclude("^(?!.*\\.class$).*$") //Consider only .class files (to avoid debug messages etc. on .dlls, etc
                            //Exclude the following: the assumption here is that no custom functionality will ever be present
                            // under these package name prefixes.
                            .exclude("^org.nd4j.*").exclude("^org.bytedeco.*") //JavaCPP
                            .exclude("^com.fasterxml.*")//Jackson
                            .exclude("^org.apache.*") //Apache commons, Spark, log4j etc
                            .exclude("^org.projectlombok.*").exclude("^com.twelvemonkeys.*")
                            .exclude("^org.joda.*").exclude("^org.slf4j.*").exclude("^com.google.*")
                            .exclude("^org.reflections.*").exclude("^ch.qos.*") //Logback
                    ).addUrls(scanUrls).setScanners(new DataVecSubTypesScanner(interfaces, classesList)));
            org.reflections.Store store = reflections.getStore();

            Iterable<String> subtypesByName = store.getAll(DataVecSubTypesScanner.class.getSimpleName(), classNames);

            Set<? extends Class<?>> subtypeClasses = Sets.newHashSet(ReflectionUtils.forNames(subtypesByName));
            subtypesClassCache = new HashSet<>();
            for (Class<?> c : subtypeClasses) {
                if (Modifier.isAbstract(c.getModifiers()) || Modifier.isInterface(c.getModifiers())) {
                    //log.info("Skipping abstract/interface: {}",c);
                    continue;
                }
                subtypesClassCache.add(c);
            }
        }

        //Second: get all currently registered subtypes for this mapper
        Set<Class<?>> registeredSubtypes = new HashSet<>();
        for (Class<?> c : classes) {
            AnnotatedClass ac = AnnotatedClass.construct(c, mapper.getSerializationConfig().getAnnotationIntrospector(),
                    null);
            Collection<NamedType> types =
                    mapper.getSubtypeResolver().collectAndResolveSubtypes(ac, mapper.getSerializationConfig(),
                            mapper.getSerializationConfig().getAnnotationIntrospector());
            for (NamedType nt : types) {
                registeredSubtypes.add(nt.getType());
            }
        }

        //Third: register all _concrete_ subtypes that are not already registered
        List<NamedType> toRegister = new ArrayList<>();
        for (Class<?> c : subtypesClassCache) {
            //Check if it's concrete or abstract...
            if (Modifier.isAbstract(c.getModifiers()) || Modifier.isInterface(c.getModifiers())) {
                //log.info("Skipping abstract/interface: {}",c);
                continue;
            }

            if (!registeredSubtypes.contains(c)) {
                String name;
                if (ClassUtils.isInnerClass(c)) {
                    Class<?> c2 = c.getDeclaringClass();
                    name = c2.getSimpleName() + "$" + c.getSimpleName();
                } else {
                    name = c.getSimpleName();
                }
                toRegister.add(new NamedType(c, name));
                if (log.isDebugEnabled()) {
                    for (Class<?> baseClass : classes) {
                        if (baseClass.isAssignableFrom(c)) {
                            log.debug("Registering class for JSON serialization: {} as subtype of {}", c.getName(),
                                    baseClass.getName());
                            break;
                        }
                    }
                }
            }
        }

        mapper.registerSubtypes(toRegister.toArray(new NamedType[toRegister.size()]));
        //Recreate the mapper (via copy), as mapper won't use registered subtypes after first use
        mapper = mapper.copy();
        return mapper;
    }

    private static ObjectMapper initMapperJson() {
        return initMapper(new JsonFactory());
    }

    private static ObjectMapper initMapperYaml() {
        return initMapper(new YAMLFactory());
    }

    private static ObjectMapper initMapper(JsonFactory factory) {
        ObjectMapper om = new ObjectMapper(factory);
        om.registerModule(new JodaModule());
        om.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        om.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        om.enable(SerializationFeature.INDENT_OUTPUT);
        om.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.NONE);
        om.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
        return om;
    }

    /**
     * Builder class for constructing a ImageTransformProcess
     */
    public static class Builder {

        private List<ImageTransform> transformList;
        private int seed = 0;

        public Builder() {
            transformList = new ArrayList<>();
        }

        public Builder seed(int seed) {
            this.seed = seed;
            return this;
        }

        public Builder cropImageTransform(int crop) {
            transformList.add(new CropImageTransform(crop));
            return this;
        }

        public Builder cropImageTransform(int cropTop, int cropLeft, int cropBottom, int cropRight) {
            transformList.add(new CropImageTransform(cropTop, cropLeft, cropBottom, cropRight));
            return this;
        }

        public Builder colorConversionTransform(int conversionCode) {
            transformList.add(new ColorConversionTransform(conversionCode));
            return this;
        }

        public Builder equalizeHistTransform(int conversionCode) {
            transformList.add(new EqualizeHistTransform(conversionCode));
            return this;
        }

        public Builder filterImageTransform(String filters, int width, int height) {
            transformList.add(new FilterImageTransform(filters, width, height));
            return this;
        }

        public Builder filterImageTransform(String filters, int width, int height, int channels) {
            transformList.add(new FilterImageTransform(filters, width, height, channels));
            return this;
        }

        public Builder flipImageTransform(int flipMode) {
            transformList.add(new FlipImageTransform(flipMode));
            return this;
        }

        public Builder randomCropTransform(int height, int width) {
            transformList.add(new RandomCropTransform(height, width));
            return this;
        }

        public Builder randomCropTransform(long seed, int height, int width) {
            transformList.add(new RandomCropTransform(seed, height, width));
            return this;
        }

        public Builder resizeImageTransform(int newWidth, int newHeight) {
            transformList.add(new ResizeImageTransform(newWidth, newHeight));
            return this;
        }

        public Builder rotateImageTransform(float angle) {
            transformList.add(new RotateImageTransform(angle));
            return this;
        }

        public Builder rotateImageTransform(float centerx, float centery, float angle, float scale) {
            transformList.add(new RotateImageTransform(centerx, centery, angle, scale));
            return this;
        }

        public Builder scaleImageTransform(float delta) {
            transformList.add(new ScaleImageTransform(delta));
            return this;
        }

        public Builder scaleImageTransform(float dx, float dy) {
            transformList.add(new ScaleImageTransform(dx, dy));
            return this;
        }

        public Builder warpImageTransform(float delta) {
            transformList.add(new WarpImageTransform(delta));
            return this;
        }

        public Builder warpImageTransform(float dx1, float dy1, float dx2, float dy2, float dx3, float dy3, float dx4, float dy4) {
            transformList.add(new WarpImageTransform(dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4));
            return this;
        }


        public ImageTransformProcess build() { return new ImageTransformProcess(this);}

    }
}
