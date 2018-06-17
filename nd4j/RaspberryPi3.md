Please follow following instructions to build nd4j for raspberry PI:

	1. {In build machine] compile libnd4j as follows:
		- $git clone https://github.com/deeplearning4j/libnd4j.git
		- For cross compilation use this link:http://stackoverflow.com/questions/19162072/installing-raspberry-pi-cross-compiler
		- make sure to use the 4.9 version of gcc (raspberrypi/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-g++)
	2.[In build machine]  build using "$./buildnativeoperations.sh -o linux-armhf"
	3. [In build machine] Install maven 3.3.9 (3.0.X does not work)
	4. [In build machine] followed instructions mentioned in https://deeplearning4j.org/buildinglocally :
			a. $export LIBND4J_HOME=<pathTond4JNI>
			
			b. Build and install nd4j to maven locally (using the forked nd4j specifically changed for raspberry pi)
				$git clone https://github.com/deeplearning4j/nd4j.git
				$cd nd4j
			
			c. $mvn clean install -Djavacpp.platform=linux-armhf -Djavacpp.platform.compiler=$HOME/raspberrypi/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-g++ -DskipTests  -Dmaven.javadoc.skip=true  -pl '!:nd4j-cuda-9.1,!:nd4j-cuda-9.1-platform,!:nd4j-tests'

	5. [In build machine ] build the source of dependant appllication with above (step 7) dependencies.
	6. [In raspbian ]copy the generated jar of dependant application to raspberry
	7. [In raspbian ]download the libraries inside the folders of build machine nd4j-backends/nd4j-backend-impls/nd4j-native/target/classes/org/nd4j/nativeblas/linux-armhf/  to a permanent folder containing libs (if possible to a system folder)
	8.[In raspbian ] export the variable LD_LIBRARY_PATH to the path set in step 10: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<the path to libraryy>
	9. [In raspbian ]java -jar myjar.jar
