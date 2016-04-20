###Running tests


1. Build cppu test from source

```
git clone https://github.com/cpputest/cpputest

```

2. To prevent cuda from complaining when compiling we need to override the memory
   leak detection macros, do this by copying
   MemoryLeakDetectorNewMacros.h to:

```
${CPPUTEST_HOME}/include/CppUTest/
```


This will override the new and delete macros to add the right compiler directives
for cuda


3. Run cmake/make

```
cmake . && make
sudo make install
```

4. Run the tests