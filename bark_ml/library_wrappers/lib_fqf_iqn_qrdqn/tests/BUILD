py_test(
    name = "save_load_test",
    srcs = ["save_load_test.py"],
    data = [
            "@bark_project//bark:generate_core",
            ],
    imports = ["../external/bark_project/bark/python_wrapper/",
              ],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/environments:gym",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
            "//bark_ml/commons:py_spaces"
           ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_loss_functions",
    srcs = ["test_loss_functions.py"],
    data = [
            "@bark_project//bark:generate_core",
            ],
    imports = ["../external/bark_project/bark/python_wrapper/",
              ],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/environments:gym",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
            "//bark_ml/commons:py_spaces",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn:utils",
           ],
    visibility = ["//visibility:public"],
)

py_library(
   name = "test_demo_behavior",
   srcs = ["test_demo_behavior.py"],
   visibility = ["//visibility:public"],
)

py_test(
    name = "demonstration_collector_test",
    srcs = [
            "demonstration_collector_test.py"
            ],
    data = [
            "@bark_project//bark:generate_core"
            ],
    imports = ["../external/bark_project/bark/python_wrapper/",
              ],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/environments:gym",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent/demonstrations:demonstrations",
            "//bark_ml/commons:py_spaces",
            ":test_demo_behavior"
           ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_training_benchmark",
    srcs = ["test_training_benchmark.py"],
    data = [
        "//bark_ml:generate_core",
        "@bark_project//bark:generate_core",
    ],
    deps = [
        "//bark_ml/environments:blueprints",
        "//bark_ml/environments:gym",
        "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
        "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:training_benchmark_database",
        "@bark_project//bark/runtime/viewer:py_viewer",
        "@benchmark_database//load:benchmark_database",
        "@benchmark_database//serialization:database_serializer",
        "@planner_uct//bark_mcts/runtime/scenario:config_readers",
        "@planner_uct//bark_mcts/runtime/scenario:scenario_set_generation",
    ],
)

py_test(
    name = "demonstrator_buffer_tests",
    srcs = ["demonstrator_buffer_tests.py"],
    data = [
            "@bark_project//bark/python_wrapper:core.so"
            ],
    imports = ["../external/bark_project/bark/python_wrapper/",
              ],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/memory:init",
            "//bark_ml/commons:py_spaces"
           ],
    visibility = ["//visibility:public"],       
)
