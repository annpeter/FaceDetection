LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

OPENCV_INSTALL_MODULES:= on
OPENCV_CAMERA_MODULES := off

OPENCV_LIB_TYPE:=SHARED

include $(LOCAL_PATH)/opencv_sdk/OpenCV.mk

LOCAL_C_INCLUDES := $(LOCAL_PATH) \
    $(LOCAL_PATH)/opencv_sdk/include/

LOCAL_SRC_FILES  := DetectionBasedTracker_jni.cpp


LOCAL_LDLIBS     += -llog -ldl

LOCAL_MODULE     := detection_based_tracker

include $(BUILD_SHARED_LIBRARY)
