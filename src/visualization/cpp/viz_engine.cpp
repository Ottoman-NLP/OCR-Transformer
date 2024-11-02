#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// OpenGL headers (correct order is important)
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// ImGui headers
#define IMGUI_IMPL_OPENGL_ES2
#define IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#include <imgui.h>
#include "imgui_impl/imgui_impl_glfw.h"
#include "imgui_impl/imgui_impl_opengl3.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <mutex>
#include <deque>
#include <string>
#include <stdexcept>

namespace py = pybind11;

// OpenGL function declarations
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

#ifndef GL_TEXTURE_WRAP_S
#define GL_TEXTURE_WRAP_S 0x2802
#endif

#ifndef GL_TEXTURE_WRAP_T
#define GL_TEXTURE_WRAP_T 0x2803
#endif

class VizEngine {
private:
    std::deque<float> train_losses;
    std::deque<float> val_losses;
    std::deque<float> accuracies;
    std::deque<float> learning_rates;
    mutable std::mutex data_mutex;
    GLFWwindow* window;
    bool initialized;
    const size_t max_points = 100;
    const int window_width = 1280;
    const int window_height = 720;
    const char* glsl_version = "#version 130";

public:
    VizEngine() : initialized(false), window(nullptr) {
        // Initialize GLFW
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        // Configure GLFW
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
        
        #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        #endif

        // Create window
        window = glfwCreateWindow(window_width, window_height, "Training Progress", nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        // Initialize GLAD
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            glfwDestroyWindow(window);
            glfwTerminate();
            throw std::runtime_error("Failed to initialize GLAD");
        }

        // Initialize ImGui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        // Setup Platform/Renderer backends
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);

        // Setup style
        ImGui::StyleColorsDark();
        
        initialized = true;
    }

    ~VizEngine() {
        if (initialized) {
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
            if (window) {
                glfwDestroyWindow(window);
            }
            glfwTerminate();
        }
    }

    void update(float train_loss, float val_loss, float accuracy, float lr) {
        if (!initialized) return;
        
        std::lock_guard<std::mutex> lock(data_mutex);
        
        train_losses.push_back(train_loss);
        val_losses.push_back(val_loss);
        accuracies.push_back(accuracy);
        learning_rates.push_back(lr);

        if (train_losses.size() > max_points) {
            train_losses.pop_front();
            val_losses.pop_front();
            accuracies.pop_front();
            learning_rates.pop_front();
        }

        render_frame();
    }

private:
    void render_frame() {
        if (!initialized || glfwWindowShouldClose(window)) return;

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Create main window
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(static_cast<float>(window_width), static_cast<float>(window_height)));
        ImGui::Begin("Training Progress", nullptr, 
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

        if (!train_losses.empty()) {
            // Convert deques to vectors for plotting
            std::vector<float> train_loss_arr(train_losses.begin(), train_losses.end());
            std::vector<float> val_loss_arr(val_losses.begin(), val_losses.end());
            std::vector<float> acc_arr(accuracies.begin(), accuracies.end());
            std::vector<float> lr_arr(learning_rates.begin(), learning_rates.end());

            // Plot metrics
            ImGui::PlotLines("Training Loss", train_loss_arr.data(), static_cast<int>(train_loss_arr.size()));
            ImGui::PlotLines("Validation Loss", val_loss_arr.data(), static_cast<int>(val_loss_arr.size()));
            ImGui::PlotLines("Accuracy", acc_arr.data(), static_cast<int>(acc_arr.size()));
            ImGui::PlotLines("Learning Rate", lr_arr.data(), static_cast<int>(lr_arr.size()));

            // Display current values
            ImGui::Text("Current Metrics:");
            ImGui::Text("Training Loss: %.4f", train_losses.back());
            ImGui::Text("Validation Loss: %.4f", val_losses.back());
            ImGui::Text("Accuracy: %.2f%%", accuracies.back());
            ImGui::Text("Learning Rate: %.6f", learning_rates.back());
        }

        ImGui::End();
        ImGui::Render();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }
};

PYBIND11_MODULE(viz_engine, m) {
    py::class_<VizEngine>(m, "VizEngine")
        .def(py::init<>())
        .def("update", &VizEngine::update);
}