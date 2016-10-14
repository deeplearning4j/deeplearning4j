package org.deeplearning4j.ui.i18n;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.ui.api.I18N;
import org.reflections.Reflections;
import org.reflections.scanners.ResourcesScanner;
import org.reflections.util.ConfigurationBuilder;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * Default internationalization implementation.
 * Content for internationalization is implemented using 2 mechanisms:<br>
 * (a) Resource files<br>
 * (b) Java classes implementing the {@link I18NContentSource} interface, loaded via reflection<br>
 * <p>
 * For resource files: they should be specified as follows:
 * 1. In the /dl4j_i18n/ directory in resources
 * 2. Filenames should be "somekey.langcode" - for example, "index.en" or "index.ja"
 * 3. Within each file: format for key "index.title" would be encoded in "index.en" as "title=Title here"
 *    For line breaks in strings: <TODO>
 * <p>
 * Loading of these UI resources is done as follows:<br>
 * - On initialization of the DefaultI18N:<br>
 * &nbsp;&nbsp;- Resource files for the default language are loaded<br>
 * &nbsp;&nbsp;- the classpath is scanned for any {@link I18NContentSource} classes<br>
 * - If a different language is requested, the content will be loaded on demand (and stored in memory for future use)
 *
 * @author Alex Black
 */
@Slf4j
public class DefaultI18N implements I18N {

    public static final String DEFAULT_LANGUAGE = "en";
    public static final String DEFAULT_I8N_RESOURCES_DIR = "dl4j_i18n";

    private static DefaultI18N instance;

    public static synchronized I18N getInstance(){
        if(instance == null) instance = new DefaultI18N();
        return instance;
    }


    private String currentLanguage = DEFAULT_LANGUAGE;

    private Set<String> loadedLanguages = Collections.synchronizedSet(new HashSet<>());

    private DefaultI18N(){

        //Load default language...
        loadLanguageResources(currentLanguage);

    }

    private synchronized void loadLanguageResources(String languageCode){
        if(loadedLanguages.contains(languageCode)) return;

        //Scan classpath for resources in the /dl4j_i18n/ directory...

        URL url = null;
        try{
            url = new File(DEFAULT_I8N_RESOURCES_DIR).toURI().toURL();
        }catch (MalformedURLException e){
            log.error("Could not load internationalization content from directory {}", DEFAULT_I8N_RESOURCES_DIR);
        }

        Reflections reflections = new Reflections(
                new ConfigurationBuilder()
                .setScanners(new ResourcesScanner())
                .setUrls()
        );


        loadedLanguages.add(languageCode);
    }

    @Override
    public String getMessage(String key) {
        return getMessage(currentLanguage, key);
    }

    @Override
    public String getMessage(String langCode, String key) {
        return null;
    }

    @Override
    public String getDefaultLanguage() {
        return currentLanguage;
    }

    @Override
    public void setDefaultLanguage(String langCode) {
        //TODO Validation
        this.currentLanguage = langCode;
    }
}
