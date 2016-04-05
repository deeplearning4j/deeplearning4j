## Developer Readme

Some things to know regarding the deeplearning4j-ui-components module:

- This module is designed with the following goals in mind:
    - Providing easy Java/JavaScript interoperability for reusable user interface components (charts, tables etc)
    - Maintainability via the use of TypeScript and proper OO design
  
The directory/package and file structure is mirrored (as much as possible) on both the java and typescript sides.

Some design principles here:

- Content and style are separate
    - The goal is to have style information as an *optional* component on the Java side
    - In principle styling can be done entirely in Java, entirely in JavaScript/CSS, or via a mixture of both (style options defined in Java will override those defined in CSS)
    - In both the Java and TypeScript, there are Style objects; these define properties such as colors or line widths
- Java objects for UI components are created; these are converted to JSON using Jackson (typically posted to a web server)
- JSON is converted back into JavaScript/TypeScript objects using the TS object constructors (i.e., these parse the JSON)
    - Consequently, any changes made to the Java UI component classes should be mirrored
- After the JS/TS versions of the UI objects are created, they can be added to an existing component using the .render(addToObject: JQuery) method
    - For charts, d3.js is used


Some additional setup that is required for developing/maintaining the TypeScript code:

- Install [node.js](https://nodejs.org/en/) and [TypeScript compiler](https://www.typescriptlang.org/#download-links)
- Then, in IntelliJ
    - Ensure auto-compilation to TypeScript is disabled (untick "Enable TypeScript Compiler" in File -> Settings -> Languages & Frameworks -> TypeScript)
    - Install the File Watchers plugin (File -> Settings -> Plugins). This will enable auto compilation to a *single* javascript library any time you make a change to the .ts files (next step)
- Set up a new file watcher for compilation
    - File -> Settings -> Tools -> File Watchers; (+) -> TypeScript
    - Program: point to typescript compiler (on Windows, for example: "C:\Users\ *UserName* \AppData\Roaming\npm\tsc.cmd")
    - Arguments: empty (empty: use tsconfig.json)

After the above setup, all TypeScript files will be compiled to a single javascript file.
(Make sure your TypeScript compiler version is up to date; tsconfig.json is only suported by version 1.5 or above) 

- Location: /deeplearning4j-ui-components/src/main/resources/assets/dl4j-ui.js
- A source map file (dl4j-ui.js.map) will also be produced; this aids with debugging (can see/debug original typescript code, even after it is converted to JavaScript)
- File name, location, and other options are defined in the tsconfig.json
  
  