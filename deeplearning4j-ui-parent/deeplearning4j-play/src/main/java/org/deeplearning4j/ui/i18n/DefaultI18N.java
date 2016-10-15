package org.deeplearning4j.ui.i18n;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.ui.api.I18N;
import org.reflections.Reflections;
import org.reflections.scanners.ResourcesScanner;
import org.reflections.util.ConfigurationBuilder;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.*;
import java.util.regex.Pattern;

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
    public static final String FALLBACK_LANGUAGE = "en";    //use this if the specified language doesn't have the requested message
    public static final String DEFAULT_I8N_RESOURCES_DIR = "dl4j_i18n";

    private static final String LINE_SEPARATOR = System.getProperty("line.separator");

    private static DefaultI18N instance;

    private Map<String,Map<String,String>> messagesByLanguage = new HashMap<>();

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

        URL url;
        try{
            url = new File("").toURI().toURL();
        }catch (MalformedURLException e){
            throw new RuntimeException(e);  //Should never happen
        }

        Reflections reflections = new Reflections(
                new ConfigurationBuilder()
                .setScanners(new ResourcesScanner())
                .setUrls(url)
        );

        String pattern = ".*" + languageCode;
        Set<String> resources = reflections.getResources(Pattern.compile(pattern));

        String regex = ".*/" + DEFAULT_I8N_RESOURCES_DIR + "/.*" + languageCode;

        Map<String,String> messages = new HashMap<>();

        for(String s : resources){
            if(!s.matches(regex)) continue;

            log.info("Attempting to parse file: {}", s);
            parseFile(new File(s), messages);
        }

        messagesByLanguage.put(languageCode, messages);

        loadedLanguages.add(languageCode);
    }

    private void parseFile(File f, Map<String,String> results){
        String str;
        try{
            str = FileUtils.readFileToString(f);
        } catch (IOException e){
            log.warn("Error parsing UI I18N content file; skipping: {}", f.getName(), e.getMessage());
            return;
        }

        //TODO need to think more carefully about how to parse this, with multi-line messages, etc
        int count = 0;
        String[] lines = str.split(LINE_SEPARATOR);
        for(String line : lines){
            if(!line.matches(".+=.*")){
                log.warn("Invalid line in I18N file: {}, \"{}\"", f, line);
                continue;
            }
            int idx = line.indexOf('=');
            String key = line.substring(0, idx);
            String value = line.substring(Math.min(idx+1,line.length()));
            results.put(key,value);
            count++;
        }

        //TODO don't log (only for development)
        log.info("Loaded {} messages from file {}",count,f);
    }

    @Override
    public String getMessage(String key) {
        return getMessage(currentLanguage, key);
    }

    @Override
    public String getMessage(String langCode, String key) {
        Map<String,String> messagesForLanguage = messagesByLanguage.get(langCode);
        if(messagesForLanguage == null){
            synchronized (this){
                //Synchronized to avoid loading multiple times in case of multi-threaded requests
                if(messagesByLanguage.get(langCode) == null){
                    loadLanguageResources(langCode);
                }
            }
            messagesForLanguage = messagesByLanguage.get(langCode);
        }

        String msg = messagesForLanguage.get(key);
        if(msg == null && !FALLBACK_LANGUAGE.equals(langCode)){
            //Try getting the result from the fallback language
            return getMessage(FALLBACK_LANGUAGE, key);
        }

        return msg;
    }

    @Override
    public String getDefaultLanguage() {
        return currentLanguage;
    }

    @Override
    public void setDefaultLanguage(String langCode) {
        //TODO Validation
        this.currentLanguage = langCode;
        log.info("UI: Set language to {}",langCode);
    }
}
