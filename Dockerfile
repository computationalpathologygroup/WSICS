FROM debian

### Acquires required packages and libraries ###
RUN apt-get update && \
	apt-get upgrade -y && \
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
	libboost-all-dev \
	libjpeg-dev \
	libtiff-dev \
	libcairo2-dev \
	libgdk-pixbuf2.0-dev \
	libxml2-dev \
	zlib1g-dev \
	swig3.0

### Compiles the required libraries ###
RUN mkdir libraries
WORKDIR /libraries/

RUN git clone https://github.com/Kitware/CMake.git && \
	ls -lh && \
	cd CMake && \
	cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/ . && \
	make -j 4 && \
	make install && \
	cd .. && \
	rm CMake -r

RUN wget http://www.ece.uvic.ca/%7Efrodo/jasper/software/jasper-2.0.14.tar.gz && \
	tar -xzf jasper-2.0.14.tar.gz && \
	rm jasper-2.0.14.tar.gz && \
	cd jasper-2.0.14/build && \
	cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/ .. && \
	make -j 4 && \
	make install && \
	cd ../.. && \
	rm jasper-2.0.14 -r

RUN wget https://github.com/uclouvain/openjpeg/archive/v2.3.0.tar.gz && \
	tar -xzf v2.3.0.tar.gz && \
	mkdir openjpeg-2.3.0/build/ && \
	rm v2.3.0.tar.gz && \
	cd openjpeg-2.3.0/build && \
	cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/ .. && \
	make -j 4 && \
	make install && \
	cd ../.. && \
	rm openjpeg-2.3.0 -r

RUN wget https://www.sqlite.org/2019/sqlite-autoconf-3270100.tar.gz && \
	tar -xzf sqlite-autoconf-3270100.tar.gz && \
	rm sqlite-autoconf-3270100.tar.gz && \
	cd sqlite-autoconf-3270100 && \
	./configure --libdir /usr/lib/ --includedir /usr/include/ && \
	make -j 4 && \
	make install && \
	cd .. && \
	rm sqlite-autoconf-3270100 -r

RUN git clone https://github.com/zeux/pugixml.git && \
	cd pugixml && \
	cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/ . && \
	make -j 4 && \
	make install && \
	cp src/* /usr/include/ && \
	cd .. && \
	rm pugixml -r

RUN	git clone https://github.com/openslide/openslide.git && \
	cd openslide && \
	autoreconf -i && \
	./configure --libdir /usr/lib/ --includedir /usr/include/ && \
	make -j 4 && \
	make install && \
	cd .. && \
	rm openslide -r

RUN	git clone https://github.com/opencv/opencv.git && \
	cd opencv && \
	mkdir build && \
	cd build && \
	cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/ .. && \
	make -j 4 && \
	make install && \
	cd ../.. && \
	rm opencv -r
	
RUN git clone --single-branch --branch master https://github.com/computationalpathologygroup/ASAP.git && \
	cd ASAP && \
	cmake \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX=/usr/ \
	-DOPENSLIDE_INCLUDE_DIR=/usr/include/openslide/ \
	-DBOOST_ROOT=/usr/ \
	-DBUILD_DIAG_PATHOLOGY_EXECUTABLES=OFF \
	-DBUILD_DIAG_PATHOLOGY_TESTS=OFF \
	-DBUILD_BUILD_EXECUTABLES=OFF \
	-DBUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT=OFF \
	-DBUILD_TESTS=OFF \
	. && \
	make -j 4 && \
	make install && \
	cd .. && \
	rm ASAP -r

### Compiles the stain normalization algorithm ###
ADD . ./WSICS/
RUN	cd WSICS && \
	cmake \
	-DCMAKE_BUILD_TYPE=Release \
	-DBOOST_ROOT=/usr/ \
	-DASAP_INCLUDE_DIRS=/usr/include/ \
	-DASAP_LIB_DIRS=/usr/lib/ \
	. && \
	make -j 4 && \
	make install && \
	cd ..  && \
	rm WSICS -r

### Sets the entrypoint ###
RUN rm /libraries/ -r
WORKDIR /usr/bin/
ENTRYPOINT [ "wsics" ]
CMD []