cmake_minimum_required(VERSION 3.8)
project(ros2-auto-sysid)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(sysid_core)
ament_python_install_package(sysid_models)
ament_python_install_package(sysid_excite)
ament_python_install_package(sysid_exporter)

install(PROGRAMS
  scripts/run_sim2sim_benchmark.py
  scripts/sysid_excitation_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  urdf
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
