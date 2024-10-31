import os
import shutil
import urllib.request

def setup_imgui():
    # Create imgui_impl directory
    os.makedirs('imgui_impl', exist_ok=True)
    
    # Copy existing header files
    vcpkg_include = r'C:\vcpkg\installed\x64-windows\include'
    header_files = ['imgui_impl_glfw.h', 'imgui_impl_opengl3.h']
    
    print("Copying existing header files...")
    for header in header_files:
        src = os.path.join(vcpkg_include, header)
        dst = os.path.join('imgui_impl', header)
        shutil.copy2(src, dst)
        print(f"Copied {header}")
    
    # Download implementation files
    cpp_files = [
        ('imgui_impl_glfw.cpp', 'https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_glfw.cpp'),
        ('imgui_impl_opengl3.cpp', 'https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3.cpp')
    ]
    
    print("\nDownloading implementation files...")
    for filename, url in cpp_files:
        target_path = os.path.join('imgui_impl', filename)
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, target_path)
        print(f"Saved to {target_path}")

if __name__ == "__main__":
    setup_imgui() 