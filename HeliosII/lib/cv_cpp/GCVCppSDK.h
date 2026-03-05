#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>
#ifdef _WIN32
#include <windows.h>
#endif

// Base class that user DLLs must implement
class CCVWorkerBase {
public:
    virtual ~CCVWorkerBase() = default;
    virtual std::pair<cv::Mat, std::vector<uint8_t>> process(cv::Mat& frame) = 0;
};

// Controller and GCV functions - resolved at runtime from CVCppWrapper.exe
// - get_actual(button): Physical controller state (0-100 buttons, -100 to +100 axes)
// - get_val(button):    Virtual controller state (reported output)
// - send_gcvdata():     Send GCV data immediately (also auto-sent on frame return)
namespace CCV {
namespace Internal {
    inline HMODULE getWrapper() {
        static HMODULE h = GetModuleHandleW(L"CVCppWrapper.exe");
        return h;
    }
    
    template<typename T>
    T getProc(const char* name) {
        static T fn = reinterpret_cast<T>(GetProcAddress(getWrapper(), name));
        return fn;
    }
}
}

inline float get_actual(int button) {
    auto fn = CCV::Internal::getProc<float(*)(int)>("get_actual");
    return fn ? fn(button) : 0.0f;
}

inline float get_val(int button) {
    auto fn = CCV::Internal::getProc<float(*)(int)>("get_val");
    return fn ? fn(button) : 0.0f;
}

inline void send_gcvdata(const uint8_t* data, size_t size) {
    auto fn = CCV::Internal::getProc<void(*)(const uint8_t*, size_t)>("send_gcvdata");
    if (fn) fn(data, size);
}

inline void send_gcvdata(const std::vector<uint8_t>& data) {
    if (!data.empty()) {
        send_gcvdata(data.data(), data.size());
    }
}

// Convenience namespace for controller constants (Gtuner IV compatible)
namespace CCV {
    // Button identifiers (values 0.00-100.00)
    constexpr int BUTTON_1 = 0;
    constexpr int BUTTON_2 = 1;
    constexpr int BUTTON_3 = 2;
    constexpr int BUTTON_4 = 3;
    constexpr int BUTTON_5 = 4;
    constexpr int BUTTON_6 = 5;
    constexpr int BUTTON_7 = 6;
    constexpr int BUTTON_8 = 7;
    constexpr int BUTTON_9 = 8;
    constexpr int BUTTON_10 = 9;
    constexpr int BUTTON_11 = 10;
    constexpr int BUTTON_12 = 11;
    constexpr int BUTTON_13 = 12;
    constexpr int BUTTON_14 = 13;
    constexpr int BUTTON_15 = 14;
    constexpr int BUTTON_16 = 15;
    constexpr int BUTTON_17 = 16;
    constexpr int BUTTON_18 = 17;
    constexpr int BUTTON_19 = 18;
    constexpr int BUTTON_20 = 19;
    constexpr int BUTTON_21 = 20;
    
    // Axis identifiers (values -100.00 to +100.00)
    constexpr int STICK_1_X = 21;
    constexpr int STICK_1_Y = 22;
    constexpr int STICK_2_X = 23;
    constexpr int STICK_2_Y = 24;
    constexpr int POINT_1_X = 25;
    constexpr int POINT_1_Y = 26;
    constexpr int POINT_2_X = 27;
    constexpr int POINT_2_Y = 28;
    constexpr int ACCEL_1_X = 29;
    constexpr int ACCEL_1_Y = 30;
    constexpr int ACCEL_1_Z = 31;
    constexpr int ACCEL_2_X = 32;
    constexpr int ACCEL_2_Y = 33;
    constexpr int ACCEL_2_Z = 34;
    constexpr int GYRO_1_X = 35;
    constexpr int GYRO_1_Y = 36;
    constexpr int GYRO_1_Z = 37;
    constexpr int PADDLE_1 = 38;
    constexpr int PADDLE_2 = 39;
    constexpr int PADDLE_3 = 40;
    constexpr int PADDLE_4 = 41;
    

}

// Macros for optional DLL exports
#define CCV_DEFAULT_DESTROY_WORKER \
extern "C" __declspec(dllexport) void destroyWorker(CCVWorkerBase* worker) { \
    delete worker; \
}

#define CCV_DEFAULT_GET_VERSION \
extern "C" __declspec(dllexport) const char* getVersion() { \
    return "1.0.0"; /* Default version */ \
}

/*
MINIMAL EXAMPLE:

#include "GCVCppSDK.h"

class MyWorker : public CCVWorkerBase {
public:
    MyWorker(int, int) {}
    
    std::pair<cv::Mat, std::vector<uint8_t>> process(cv::Mat& frame) override {
        if (get_actual(CCV::BUTTON_16) > 50.0f) {
            cv::circle(frame, cv::Point(100, 100), 30, cv::Scalar(0, 255, 0), -1);
        }
        
        std::vector<uint8_t> gcvData = {1, 2, 3};
        return {frame, gcvData};
    }
};

extern "C" __declspec(dllexport) CCVWorkerBase* createWorker(int w, int h) {
    return new MyWorker(w, h);
}

CCV_DEFAULT_DESTROY_WORKER
CCV_DEFAULT_GET_VERSION
*/