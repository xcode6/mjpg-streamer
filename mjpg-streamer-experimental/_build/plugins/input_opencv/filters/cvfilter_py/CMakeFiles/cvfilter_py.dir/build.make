# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pi/mjpg-streamer/mjpg-streamer-experimental

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pi/mjpg-streamer/mjpg-streamer-experimental/_build

# Include any dependencies generated for this target.
include plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/depend.make

# Include the progress variables for this target.
include plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/progress.make

# Include the compile flags for this target's objects.
include plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/flags.make

plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/filter_py.cpp.o: plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/flags.make
plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/filter_py.cpp.o: ../plugins/input_opencv/filters/cvfilter_py/filter_py.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/mjpg-streamer/mjpg-streamer-experimental/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/filter_py.cpp.o"
	cd /home/pi/mjpg-streamer/mjpg-streamer-experimental/_build/plugins/input_opencv/filters/cvfilter_py && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cvfilter_py.dir/filter_py.cpp.o -c /home/pi/mjpg-streamer/mjpg-streamer-experimental/plugins/input_opencv/filters/cvfilter_py/filter_py.cpp

plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/filter_py.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cvfilter_py.dir/filter_py.cpp.i"
	cd /home/pi/mjpg-streamer/mjpg-streamer-experimental/_build/plugins/input_opencv/filters/cvfilter_py && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pi/mjpg-streamer/mjpg-streamer-experimental/plugins/input_opencv/filters/cvfilter_py/filter_py.cpp > CMakeFiles/cvfilter_py.dir/filter_py.cpp.i

plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/filter_py.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cvfilter_py.dir/filter_py.cpp.s"
	cd /home/pi/mjpg-streamer/mjpg-streamer-experimental/_build/plugins/input_opencv/filters/cvfilter_py && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pi/mjpg-streamer/mjpg-streamer-experimental/plugins/input_opencv/filters/cvfilter_py/filter_py.cpp -o CMakeFiles/cvfilter_py.dir/filter_py.cpp.s

plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/filter_py.cpp.o.requires:

.PHONY : plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/filter_py.cpp.o.requires

plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/filter_py.cpp.o.provides: plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/filter_py.cpp.o.requires
	$(MAKE) -f plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/build.make plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/filter_py.cpp.o.provides.build
.PHONY : plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/filter_py.cpp.o.provides

plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/filter_py.cpp.o.provides.build: plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/filter_py.cpp.o


plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/conversion.cpp.o: plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/flags.make
plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/conversion.cpp.o: ../plugins/input_opencv/filters/cvfilter_py/conversion.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/mjpg-streamer/mjpg-streamer-experimental/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/conversion.cpp.o"
	cd /home/pi/mjpg-streamer/mjpg-streamer-experimental/_build/plugins/input_opencv/filters/cvfilter_py && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cvfilter_py.dir/conversion.cpp.o -c /home/pi/mjpg-streamer/mjpg-streamer-experimental/plugins/input_opencv/filters/cvfilter_py/conversion.cpp

plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/conversion.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cvfilter_py.dir/conversion.cpp.i"
	cd /home/pi/mjpg-streamer/mjpg-streamer-experimental/_build/plugins/input_opencv/filters/cvfilter_py && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pi/mjpg-streamer/mjpg-streamer-experimental/plugins/input_opencv/filters/cvfilter_py/conversion.cpp > CMakeFiles/cvfilter_py.dir/conversion.cpp.i

plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/conversion.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cvfilter_py.dir/conversion.cpp.s"
	cd /home/pi/mjpg-streamer/mjpg-streamer-experimental/_build/plugins/input_opencv/filters/cvfilter_py && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pi/mjpg-streamer/mjpg-streamer-experimental/plugins/input_opencv/filters/cvfilter_py/conversion.cpp -o CMakeFiles/cvfilter_py.dir/conversion.cpp.s

plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/conversion.cpp.o.requires:

.PHONY : plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/conversion.cpp.o.requires

plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/conversion.cpp.o.provides: plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/conversion.cpp.o.requires
	$(MAKE) -f plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/build.make plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/conversion.cpp.o.provides.build
.PHONY : plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/conversion.cpp.o.provides

plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/conversion.cpp.o.provides.build: plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/conversion.cpp.o


# Object files for target cvfilter_py
cvfilter_py_OBJECTS = \
"CMakeFiles/cvfilter_py.dir/filter_py.cpp.o" \
"CMakeFiles/cvfilter_py.dir/conversion.cpp.o"

# External object files for target cvfilter_py
cvfilter_py_EXTERNAL_OBJECTS =

plugins/input_opencv/filters/cvfilter_py/cvfilter_py.so: plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/filter_py.cpp.o
plugins/input_opencv/filters/cvfilter_py/cvfilter_py.so: plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/conversion.cpp.o
plugins/input_opencv/filters/cvfilter_py/cvfilter_py.so: plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/build.make
plugins/input_opencv/filters/cvfilter_py/cvfilter_py.so: /usr/local/lib/libopencv_face.so.3.4.3
plugins/input_opencv/filters/cvfilter_py/cvfilter_py.so: /usr/local/lib/libopencv_highgui.so.3.4.3
plugins/input_opencv/filters/cvfilter_py/cvfilter_py.so: /usr/local/lib/libopencv_videoio.so.3.4.3
plugins/input_opencv/filters/cvfilter_py/cvfilter_py.so: /usr/lib/arm-linux-gnueabihf/libpython3.5m.so
plugins/input_opencv/filters/cvfilter_py/cvfilter_py.so: /usr/local/lib/libopencv_objdetect.so.3.4.3
plugins/input_opencv/filters/cvfilter_py/cvfilter_py.so: /usr/local/lib/libopencv_photo.so.3.4.3
plugins/input_opencv/filters/cvfilter_py/cvfilter_py.so: /usr/local/lib/libopencv_video.so.3.4.3
plugins/input_opencv/filters/cvfilter_py/cvfilter_py.so: /usr/local/lib/libopencv_imgcodecs.so.3.4.3
plugins/input_opencv/filters/cvfilter_py/cvfilter_py.so: /usr/local/lib/libopencv_imgproc.so.3.4.3
plugins/input_opencv/filters/cvfilter_py/cvfilter_py.so: /usr/local/lib/libopencv_core.so.3.4.3
plugins/input_opencv/filters/cvfilter_py/cvfilter_py.so: plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pi/mjpg-streamer/mjpg-streamer-experimental/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library cvfilter_py.so"
	cd /home/pi/mjpg-streamer/mjpg-streamer-experimental/_build/plugins/input_opencv/filters/cvfilter_py && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cvfilter_py.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/build: plugins/input_opencv/filters/cvfilter_py/cvfilter_py.so

.PHONY : plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/build

plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/requires: plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/filter_py.cpp.o.requires
plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/requires: plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/conversion.cpp.o.requires

.PHONY : plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/requires

plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/clean:
	cd /home/pi/mjpg-streamer/mjpg-streamer-experimental/_build/plugins/input_opencv/filters/cvfilter_py && $(CMAKE_COMMAND) -P CMakeFiles/cvfilter_py.dir/cmake_clean.cmake
.PHONY : plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/clean

plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/depend:
	cd /home/pi/mjpg-streamer/mjpg-streamer-experimental/_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pi/mjpg-streamer/mjpg-streamer-experimental /home/pi/mjpg-streamer/mjpg-streamer-experimental/plugins/input_opencv/filters/cvfilter_py /home/pi/mjpg-streamer/mjpg-streamer-experimental/_build /home/pi/mjpg-streamer/mjpg-streamer-experimental/_build/plugins/input_opencv/filters/cvfilter_py /home/pi/mjpg-streamer/mjpg-streamer-experimental/_build/plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : plugins/input_opencv/filters/cvfilter_py/CMakeFiles/cvfilter_py.dir/depend

