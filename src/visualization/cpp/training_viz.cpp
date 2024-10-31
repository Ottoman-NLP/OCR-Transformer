#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <vector>
#include <mutex>
#include <thread>

namespace py = pybind11;

class TrainingVisualizer {
private:
    GLFWwindow* window;
    std::vector<float> train_losses;
    std::vector<float> val_losses;
    std::vector<float> accuracies;
    std::vector<float> learning_rates;
    std::mutex data_mutex;
    bool running;
    std::thread render_thread;

public:
    TrainingVisualizer() : running(false) {
        // Initialize GLFW and OpenGL
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        // Create window with graphics context
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        window = glfwCreateWindow(1280, 720, "Training Progress", NULL, NULL);
        if (window == NULL) {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }

        // Initialize ImGui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 130");

        // Start render thread
        running = true;
        render_thread = std::thread(&TrainingVisualizer::render_loop, this);
    }

    ~TrainingVisualizer() {
        running = false;
        if (render_thread.joinable()) {
            render_thread.join();
        }
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void update(float train_loss, float val_loss, float accuracy, float lr) {
        std::lock_guard<std::mutex> lock(data_mutex);
        train_losses.push_back(train_loss);
        val_losses.push_back(val_loss);
        accuracies.push_back(accuracy);
        learning_rates.push_back(lr);
    }

private:
    void render_loop() {
        glfwMakeContextCurrent(window);
        while (running && !glfwWindowShouldClose(window)) {
            glfwPollEvents();
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // Render plots
            ImGui::Begin("Training Progress");
            {
                std::lock_guard<std::mutex> lock(data_mutex);
                if (!train_losses.empty()) {
                    // Plot losses
                    ImGui::PlotLines("Loss", train_losses.data(), train_losses.size(),
                                   0, "Training Loss", FLT_MAX, FLT_MAX,
                                   ImVec2(0, 80));
                    ImGui::PlotLines("Validation Loss", val_losses.data(), val_losses.size(),
                                   0, nullptr, FLT_MAX, FLT_MAX,
                                   ImVec2(0, 80));
                    
                    // Plot accuracy
                    ImGui::PlotLines("Accuracy", accuracies.data(), accuracies.size(),
                                   0, nullptr, 0.0f, 1.0f,
                                   ImVec2(0, 80));
                    
                    // Plot learning rate
                    ImGui::PlotLines("Learning Rate", learning_rates.data(), learning_rates.size(),
                                   0, nullptr, FLT_MAX, FLT_MAX,
                                   ImVec2(0, 80));
                }
            }
            ImGui::End();

            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);
        }
    }
};

PYBIND11_MODULE(training_viz_cpp, m) {
    py::class_<TrainingVisualizer>(m, "TrainingVisualizer")
        .def(py::init<>())
        .def("update", &TrainingVisualizer::update);
} 