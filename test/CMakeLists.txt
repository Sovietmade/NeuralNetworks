include_directories ("${PROJECT_SOURCE_DIR}/include")
add_executable( LogicalElement LogicalElement.cpp)
target_link_libraries(LogicalElement NeuralNetwork)

