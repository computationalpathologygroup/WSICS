CMAKE_MINIMUM_REQUIRED(VERSION 3.9)

PROJECT(stain_normalization)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}\ /std:c++latest)

SET(GROUP_BLOB_OPERATIONS
	BLOB_Operations/BLOB.h
	BLOB_Operations/BLOB_Operations.h
	BLOB_Operations/BLOB_Window.h
	BLOB_Operations/BLOB.cpp
	BLOB_Operations/BLOB_Operations.cpp
	BLOB_Operations/BLOB_Window.cpp
)
SET(GROUP_CLASSIFIER
	Classifier/Classifier.h
	Classifier/NaiveBayesClassifier.h
	Classifier/NaiveBayesFeatureClassifier.h
	Classifier/Classifier.cpp
	Classifier/NaiveBayesClassifier.cpp
	Classifier/NaiveBayesFeatureClassifier.cpp
)
SET(GROUP_HE_STAINING
	HE_Staining/HE_Classifier.h
	HE_Staining/MaskGeneration.h
	HE_Staining/HE_Classifier.cpp
	HE_Staining/MaskGeneration.cpp
)
SET(GROUP_HOUGH_TRANSFORM
	HoughTransform/AveragedEllipseParameters.h
	HoughTransform/Ellipse.h
	HoughTransform/IAccumulator.h
	HoughTransform/Line.h
	HoughTransform/LocationCell.h
	HoughTransform/PointCollection.h
	HoughTransform/RandomizedHoughTransform.h
	HoughTransform/TreeAccumulator.h
	HoughTransform/WindowedTripletDetector.h
	HoughTransform/AveragedEllipseParameters.cpp
	HoughTransform/Ellipse.cpp
	HoughTransform/Line.cpp
	HoughTransform/LocationCell.cpp
	HoughTransform/PointCollection.cpp
	HoughTransform/RandomizedHoughTransform.cpp
	HoughTransform/TreeAccumulator.cpp
	HoughTransform/WindowedTripletDetector.cpp
)
SET(GROUP_HSD
	HSD/BackgroundMask.h
	HSD/HSD_Model.h
	HSD/Transformations.h
	HSD/BackgroundMask.cpp
	HSD/HSD_Model.cpp
	HSD/Transformations.cpp
)
SET(GROUP_IO
	IO/CommandLineInterface.h
	IO/Logging/LogHandler.h
	IO/Logging/LogLevel.h
	IO/CommandLineInterface.cpp
	IO/Logging/LogHandler.cpp
	IO/Logging/LogLevel.cpp
)
SET(GROUP_MISC 
	Misc/LevelReading.h
	Misc/MiscFunctionality.h
	Misc/MiscMatrixOperations.h
	Misc/LevelReading.cpp
	Misc/MiscFunctionality.cpp
	Misc/MiscMatrixOperations.cpp
)
SET(GROUP_STANDARDIZATION
	Standardization/CxCyWeights.h
	Standardization/NormalizedLutCreation.h
	Standardization/NormalizedOutput.h
	Standardization/PixelClassificationHE.h
	Standardization/SlideStandardizationCLI.h
	Standardization/Standardization.h
	Standardization/StandardizationParameters.h
	Standardization/TransformCxCyDensity.h
	Standardization/CxCyWeights.cpp
	Standardization/NormalizedLutCreation.cpp
	Standardization/NormalizedOutput.cpp
	Standardization/PixelClassificationHE.cpp
	Standardization/SlideStandardizationCLI.cpp
	Standardization/Standardization.cpp
	Standardization/TransformCxCyDensity.cpp
)

ADD_EXECUTABLE(stain_normalization
	Main.cpp
	${GROUP_BLOB_OPERATIONS}
	${GROUP_CLASSIFIER}
	${GROUP_HE_STAINING}
	${GROUP_HOUGH_TRANSFORM}
	${GROUP_HSD}
	${GROUP_IO}
	${GROUP_MISC}
	${GROUP_STANDARDIZATION}
)

SOURCE_GROUP("BLOB_Operations"	FILES ${GROUP_BLOB_OPERATIONS})
SOURCE_GROUP("Classifier"		FILES ${GROUP_CLASSIFIER})
SOURCE_GROUP("HE_Staining"		FILES ${GROUP_HE_STAINING})
SOURCE_GROUP("HoughTransform"	FILES ${GROUP_HOUGH_TRANSFORM})
SOURCE_GROUP("HSD"				FILES ${GROUP_HSD})
SOURCE_GROUP("IO"				FILES ${GROUP_IO})
SOURCE_GROUP("Misc"				FILES ${GROUP_MISC})
SOURCE_GROUP("Standardization"	FILES ${GROUP_STANDARDIZATION})

FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem program_options system)
IF(Boost_FOUND)
	INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
	LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
ENDIF()

FIND_PACKAGE(OpenCV REQUIRED)
IF(OpenCV_FOUND)
	INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
	LINK_DIRECTORIES(${OpenCV_LIBRARY_DIRS})
ENDIF()

SET(ASAP_INCLUDE_DIRS "ASAP_INCLUDE_DIRECTORY" CACHE FILEPATH "The path to the ASAP include directory.")
SET(ASAP_LIB_DIRS "ASAP_LIB_DIRECTORY" CACHE FILEPATH "The path to the ASAP library directory.")
#if(ENV{ASAP_INCLUDE_DIRS} AND ENV{ASAP_LIB_DIRS})
	INCLUDE_DIRECTORIES(${ASAP_INCLUDE_DIRS})
	LINK_DIRECTORIES(${ASAP_LIB_DIRS})
	SET(ASAP_LIBRARIES ${ASAP_LIB_DIRS}/core.lib)
#endif()

TARGET_LINK_LIBRARIES(stain_normalization ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${ASAP_LIBRARIES})



install(TARGETS stain_normalization DESTINATION bin)