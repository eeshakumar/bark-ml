workspace(name = "bark_ml")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

# loading deps of BARK-ML
load("//utils:dependencies.bzl", "barkml_deps")
barkml_deps()

# proper load boost
load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()
