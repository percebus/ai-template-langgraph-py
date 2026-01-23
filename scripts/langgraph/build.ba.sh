#!/bin/bash

set -e
set -x

# SRC: https://docs.langchain.com/langsmith/deploy-standalone-server#docker-compose
langgraph build -t my-langgraph-api

set +x
set +e
