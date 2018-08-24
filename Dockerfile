FROM debian

### Acquires required packages and libraries ###
RUN apt-get update && \
	apt-get install -y \
	build-essential \
	cmake \
	git \
	wget \
	unzip \
	openslide-tools \
	libpcre++-dev \
	qt5-default \
	autoconf \
	automake \
	libtool \
	pkg-config \
	glib2.0 \
	libcairo2-dev \
	libgdk-pixbuf2.0-dev \
	libxml2-dev

### Compiles the required libraries ###
RUN mkdir libraries WSICS
WORKDIR /libraries/

RUN git clone https://github.com/Kitware/CMake.git && \
	ls -lh && \
	cd CMake && \
	cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local . && \
	make -j 4 && \
	make install && \
	cd .. && \
	rm CMake -r

RUN wget https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.gz && \
	tar -xzf boost_1_67_0.tar.gz && \
	rm boost_1_67_0.tar.gz && \
	cd boost_1_67_0 && \
	./bootstrap.sh --with-libraries=program_options,filesystem,system,regex,date_time,thread,chrono,atomic && \
	./b2 link=shared runtime-link=shared && \
	./b2 install && \
	cd .. && \
	rm boost_1_67_0 -r
# --prefix=/usr/local
RUN	wget ftp://dicom.offis.de/pub/dicom/offis/software/dcmtk/dcmtk363/dcmtk-3.6.3.tar.gz && \
	tar -xzf dcmtk-3.6.3.tar.gz && \
	rm dcmtk-3.6.3.tar.gz && \
	cd dcmtk-3.6.3 && \
	cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local . && \
	make -j 4 && \
	make install && \
	cd .. && \
	rm dcmtk-3.6.3 -r
	
RUN	git clone https://github.com/opencv/opencv.git && \
	cd opencv && \
	mkdir build && \
	cd build && \
	cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
	make -j 4 && \
	make install && \
	cd ../.. && \
	rm opencv -r
	
RUN wget http://download.osgeo.org/libtiff/tiff-4.0.9.tar.gz && \
	tar -xzf tiff-4.0.9.tar.gz && \
	rm tiff-4.0.9.tar.gz && \
	cd tiff-4.0.9 && \
	cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local . && \
	make -j 4 && \
	make install && \
	cd .. && \
	rm tiff-4.0.9 -r

RUN wget http://www.ijg.org/files/jpegsrc.v9c.tar.gz && \
	tar -xzf jpegsrc.v9c.tar.gz && \
	rm jpegsrc.v9c.tar.gz && \
	cd jpeg-9c && \
	./configure && \
	make -j 4 && \
	make install && \
	cd .. && \
	rm jpeg-9c -r

RUN wget http://www.zlib.net/zlib-1.2.11.tar.gz && \
	tar -xzf zlib-1.2.11.tar.gz && \
	rm zlib-1.2.11.tar.gz  && \
	cd zlib-1.2.11 && \
	cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local . && \
	make -j 4 && \
	make install && \
	cd .. && \
	rm zlib-1.2.11 -r

RUN wget https://netcologne.dl.sourceforge.net/project/swig/swig/swig-3.0.12/swig-3.0.12.tar.gz && \
	tar -xzf swig-3.0.12.tar.gz && \
	rm swig-3.0.12.tar.gz && \
	cd swig-3.0.12 && \
	./configure && \
	make -j 4 && \
	make install && \
	cd .. && \
	rm swig-3.0.12 -r
	
RUN git clone https://github.com/zeux/pugixml.git && \
	cd pugixml && \
	cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local . && \
	make -j 4 && \
	make install && \
	cp src/* /usr/local/include/ && \
	cd .. && \
	rm pugixml -r

RUN wget http://www.ece.uvic.ca/%7Efrodo/jasper/software/jasper-2.0.14.tar.gz && \
	tar -xzf jasper-2.0.14.tar.gz && \
	rm jasper-2.0.14.tar.gz && \
	cd jasper-2.0.14/build && \
	cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
	make -j 4 && \
	make install && \
	cd ../.. && \
	rm jasper-2.0.14 -r

RUN wget https://github.com/uclouvain/openjpeg/archive/v2.3.0.tar.gz && \
	tar -xzf v2.3.0.tar.gz && \
	mkdir openjpeg-2.3.0/build/ && \
	rm v2.3.0.tar.gz && \
	cd openjpeg-2.3.0/build && \
	cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
	make -j 4 && \
	make install && \
	cd ../.. && \
	rm openjpeg-2.3.0 -r

RUN wget https://www.sqlite.org/snapshot/sqlite-snapshot-201807272333.tar.gz && \
	tar -xzf sqlite-snapshot-201807272333.tar.gz && \
	cd sqlite-snapshot-201807272333 && \
	./configure && \
	make -j 4 && \
	make install && \
	cd .. && \
	rm sqlite-snapshot-201807272333 -r

RUN	git clone https://github.com/openslide/openslide.git && \
	cd openslide && \
	autoreconf -i && \
	./configure && \
	make -j 4 && \
	make install && \
	cd .. && \
	rm openslide -r
	
RUN git clone https://github.com/computationalpathologygroup/ASAP.git && \
	cd ASAP && \
	cmake \
		-D CMAKE_BUILD_TYPE=Release \
		-D CMAKE_INSTALL_PREFIX=/usr/local \
		-D OPENSLIDE_INCLUDE_DIR=/usr/local/include/openslide/ \
		-D BOOST_ROOT=/usr/local \
		. && \
	make -j 4 && \
	make install && \
	cd .. && \
	rm ASAP -r
	
ADD ./SlideStandardization .
RUN	cmake -D CMAKE_BUILD_TYPE=Release -D BOOST_ROOT=/usr/local . && \
	make -j 4 VERBOSE=1

### Compiles the stain normalization algorithm ###

### Cleans up all junk ###