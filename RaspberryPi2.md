Please follow following instructions to build nd4j on raspberry PI: 

		1. {In build machine] compile libnd4j as follows:
			- $git clone https://github.com/dschowta/libnd4j.git
			- For cross compilation use this link:http://stackoverflow.com/questions/19162072/installing-raspberry-pi-cross-compiler
			- make sure to use the 4.9 version of gcc (raspberrypi/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-g++)
		3.[In build machine]  build using "$./buildnativeoperations.sh -o linux-armhf"
		4. [In build machine] Install maven 3.3.9 (3.0.X does not work)
		5. [In build machine] followed instructions mentioned in https://deeplearning4j.org/buildinglocally :
				a. $export LIBND4J_HOME=<pathTond4JNI>
				
				b. Build and install nd4j to maven locally (using the forked nd4j specifically changed for raspberry pi)
					$git clone https://github.com/dschowta/nd4j.git
					$cd nd4j
				c. Edit  nd4j-backends\nd4j-backend-impls\nd4j-native\src\main\resources\org\bytedeco\javacpp\properties to update the paths of native toolchains
				
				d. $mvn clean  install -Djavacpp.platform=linux-armhf  -DskipTests  -Dmaven.javadoc.skip=true  -pl '!:nd4j-cuda-8.0,!:nd4j-cuda-8.0-platform,!:nd4j-tests'

		7. [In build machine ] build the source of dependant appllication with above (step 7) dependencies.
		8. [In raspbian ]copy the generated jar of dependant application to raspberry
		9. [In raspbian ]download the libraries inside the folders of build machine nd4j-backends/nd4j-backend-impls/nd4j-native/target/classes/org/nd4j/nativeblas/linux-armhf/  to a permanent folder containing libs (if possible to a system folder)
		10.[In raspbian ] export the variable LD_LIBRARY_PATH to the path set in step 10: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<the path to libraryy>
		11. [In raspbian ]java -jar myjar.jar
