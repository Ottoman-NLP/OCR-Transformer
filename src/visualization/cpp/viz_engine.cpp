#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-include-dirs"
#include <GL/gl.h>
#define IMGUI_IMPL_API
#include <imgui.h>
#define GLFW_INCLUDE_NONE
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <GLFW/glfw3.h>
#include "C:/vcpkg/installed/x64-windows/include/imgui.h"
#include "C:/vcpkg/installed/x64-windows/include/imgui_impl_glfw.h"
#include "C:/vcpkg/installed/x64-windows/include/imgui_impl_opengl3.h"
#include <vector>
#include <mutex>

#pragma clang diagnostic pop

namespace py = pybind11;

class VizEngine {
private:
    std::vector<float> train_losses;
    std::vector<float> val_losses;
    std::vector<float> accuracies;
    std::vector<float> learning_rates;
    mutable std::mutex data_mutex;
    GLFWwindow* window;
    bool initialized;

public:
    VizEngine() : initialized(false) {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
        
        // Create window
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        window = glfwCreateWindow(1280, 720, "Training Visualization", NULL, NULL);
        if (!window) {
            glfwTerminate();
            throw std::runtime_error("Failed to create window");
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        // Initialize ImGui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 130");

        initialized = true;
    }

    ~VizEngine() {
        if (initialized) {
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
            glfwDestroyWindow(window);
            glfwTerminate();
        }
    }

    void update(float train_loss, float val_loss, float accuracy, float lr) {
        std::lock_guard<std::mutex> lock(data_mutex);
        train_losses.push_back(train_loss);
        val_losses.push_back(val_loss);
        accuracies.push_back(accuracy);
        learning_rates.push_back(lr);
    }

    py::tuple get_data() const {
        std::lock_guard<std::mutex> lock(data_mutex);
        return py::make_tuple(train_losses, val_losses, accuracies, learning_rates);
    }

    size_t get_size() const {
        std::lock_guard<std::mutex> lock(data_mutex);
        return train_losses.size();
    }
};

PYBIND11_MODULE(viz_engine, m) {
    py::class_<VizEngine>(m, "VizEngine")
        .def(py::init<>())
        .def("update", &VizEngine::update)
        .def("get_data", &VizEngine::get_data)
        .def("get_size", &VizEngine::get_size);
}

#pragma clang diagnostic pop 