
# Deeplearning4j UI (deeplearning4j-play) Developer Readme

The deeplearning4j UI utilizes an embedded Play server for hosting web pages. This guide documents some basic information
about the design, and how new pages can be added.

## Overview

As of DL4J 0.7.2, version 2.4 is utilized for scala 2.10 compatibility, which was dropped in Play 2.5 and later.
Due to DL4J being managed by Maven, not all features of Play (such as conf/routes approach to routing configuration, and
the default internationalization mechanism) are supported.

Some key concepts:

- The actual HTML pages are defined in the main/views/org.deeplearning4j.ui.views.* directory
    - These views are based on Twirl template engine: in practice, templates are mostly HTML with a
    little scala code mixed in
    - The views themselves need to be rendered to a scala class (under main/scala). This is currently done manually,
    see the 'building templates' section.
- Routing is somewhat non-standard in DL4J, compared to a typical Play application
    - The DL4J has the concept of a 'UI module': a related set of pages to serve via the UI
    - Each module implements the UIModule interface, which defines a set of routes, and how to handle page requests
    - Custom modules are supported: this is not enabled by default. See PlayUIServer UI_CUSTOM_MODULE_PROPERTY for details.
- Internationalization (i.e., multiple languages) is supported, but not using the standard Play mechanisms
- The main class and entry point is PlayUIServer

## Building Templates

The current setup (using Maven + Play) means that the HTML templates need to be built manually.
Just run ```mvn compile -P buildUiTemplates``` on the command line to do this - this will generate the Scala classes
from all of the HTML files (in the views directory) and copy them to the appropriate location in the scala directory.

## Adding New Pages

Adding a new (built-in) page to the UI can be done using the following approach:

- Define a new module in org.deeplearning4j.ui.module, that implements the UIModule interface
    - Provide routes: these are the endpoints that the user will be able to query, and define what methods will be used
     to get the result to return for that page. See for example TrainModule for details
    - Each route needs to specify the method to call to get the result (must return a play.mvc.Result object) 
    - Supplier: used to return results with no args; Function: 1 arg; BiFunction and Function3: 2 and 3 args respectively.
      For function/bifunction etc, the arguments are specified with semicolons, like "/myRoute/:myArg"; the called
      method should have an appropriate number of arguments to match this.
    - Optionally: add code to handle callbacks, storage events, stats storage. See the section below.
- Add the module to the others, in the PlayUIServer. Should be 1 line: ```uiModules.add(new MyNewModule());```

## Assets

Again (due to issues with Maven + Play support), DL4J defines a custom Assets serving system. Assets are static files
such as images, CSS, javascript files, etc.
The contents of /resources/deeplearning4jUiAssets/* get mapped to /assets/* on the running server. To add a new asset,
simply add it to the assets folder, and then reference it as normal in the HTML templates or elsewhere.

## StatsStorage, Callbacks, and Getting Network and Other Info from DL4J 

The Deeplearning4j UI supports callbacks from StatsStorage. StatsStorage instances are objects/interfaces that store
training information - some of it may be static/stored, some of it may be streaming in, in real-time.

When a user calls ```UIServer.getInstance().attach(StatsStorage)``` the provided StatsStorage instance will provide
callbacks to the UI whenever something changes. For example, new information from a trained network is added to the
StatsStorage from the StatsListener, the modules that are registered for callbacks (of that type) will be notified.
The UI modules can then query the StatStorage instance (or not) to get the relevant information for displaying in the UI.

Each UI module specifies the callbacks it wants to receive via Strings (UIModule.getCallbackTypeIDs). Each String is a
key that must match the TypeID of the data to listen for; consequently, there is a correspondence between the TypeID of
the generating class (StatsListener, for example) and the TypeID that the UI module wants to receive. UI modules may
specify zero or more callback type IDs. Any information that is not relevant to the UI module (i.e., doesn't match a
specified TypeID for the module) won't be forwarded on to the UI module.


## Internationalization

The Play framework does provide an internationalization mechanism, though this was found to be problematic when running
 Play via a Maven project. Consequently, DL4J defines its own I18N implementation, that is functionally similar to the
 Play mechanism.

The I18N interface, and the DefaultI18N class define getMessage(String) methods. For example, getMessage() 
Currently (as of 0.7.2), the language setting is *server side* not client side - thus only one language can be shown per
 UI server. Furthermore, the implementation is currently set to fall back to English. 
 
The actual content for internationalization is present under the resources/dl4j_i18n directory. Each file within this
 directory should contain an ISO639 language code (en, ja, de etc) as the extension.
Conceptually, the files may be separated; in practice, the contents of all files (for the relevant language) are
In practice, just add a snippet such as ```@i18n.getMessage("train.pagetitle")``` to the HTML template to get the
appropriate entry for the current language. See the train module UI pages for more examples.

Note also that it is necessary to provide an I18N instance to the templates. See the TrainModule routing section
for an example on how to do this.
