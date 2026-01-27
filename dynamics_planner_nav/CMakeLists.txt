cmake_minimum_required(VERSION 3.10...3.26)
project(dynamics_planner_nav)

# Suppress new CMake dev warning about deprecated FindPython* in catkin on
# newer CMake versions, without affecting older versions.
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

# Prefer target-level standard; keep global for catkin compatibility
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    sensor_msgs
    move_base_msgs
    visualization_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
    costmap_2d
    teb_local_planner
    base_local_planner
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(gazebo REQUIRED)

catkin_package(
    CATKIN_DEPENDS
        roscpp
        std_msgs
        geometry_msgs
        nav_msgs
        sensor_msgs
        move_base_msgs
        visualization_msgs
        tf2
        tf2_ros
        tf2_geometry_msgs
        costmap_2d
        teb_local_planner
        base_local_planner
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Add catkin library directories (includes TEB's g2o)
link_directories(${catkin_LIBRARY_DIRS})

## Collect source files (track directory changes)
# CONFIGURE_DEPENDS makes CMake re-scan when files are added/removed
file(GLOB ROBOT_SOURCES CONFIGURE_DEPENDS src/robot/*.cpp)
file(GLOB LOCAL_PLANNER_SOURCES CONFIGURE_DEPENDS src/localPlanners/*.cpp)
file(GLOB GLOBAL_PLANNER_SOURCES CONFIGURE_DEPENDS src/globalPlanners/*.cpp)
file(GLOB UTILS_SOURCES CONFIGURE_DEPENDS src/utils/*.cpp)

# Exclude collision_publisher.cpp from UTILS_SOURCES (it's a separate executable)
list(REMOVE_ITEM UTILS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/collision_publisher.cpp)


## Build main executable
add_executable(${PROJECT_NAME}_node
    src/programs/RunMP.cpp
    ${ROBOT_SOURCES}
    ${LOCAL_PLANNER_SOURCES}
    ${GLOBAL_PLANNER_SOURCES}
    ${UTILS_SOURCES}
)

target_link_libraries(${PROJECT_NAME}_node
    ${catkin_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    ${OpenCV_LIBS}
    pthread
)

# Modern CMake target properties
target_compile_features(${PROJECT_NAME}_node PRIVATE cxx_std_17)

# Utils sources are included directly in the target above

add_dependencies(${PROJECT_NAME}_node
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
)

## Build SwitchMP executable (dynamic planner switching)
add_executable(${PROJECT_NAME}_switch_node
    src/programs/SwitchMP.cpp
    ${ROBOT_SOURCES}
    ${LOCAL_PLANNER_SOURCES}
    ${GLOBAL_PLANNER_SOURCES}
    ${UTILS_SOURCES}
)

target_link_libraries(${PROJECT_NAME}_switch_node
    ${catkin_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    ${OpenCV_LIBS}
    pthread
)

target_compile_features(${PROJECT_NAME}_switch_node PRIVATE cxx_std_17)

add_dependencies(${PROJECT_NAME}_switch_node
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
)

## Build collision_check_publisher_node executable
add_executable(collision_check_publisher_node
    src/utils/collision_publisher.cpp
)

target_link_libraries(collision_check_publisher_node
    ${catkin_LIBRARIES}
    ${GAZEBO_LIBRARIES}
)

target_compile_features(collision_check_publisher_node PRIVATE cxx_std_17)

add_dependencies(collision_check_publisher_node
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
)
