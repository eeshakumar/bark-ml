py_test(
    name = "save_load_test",
    srcs = ["save_load_test.py"],
    data = [
            "@bark_project//bark/python_wrapper:core.so"
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
