#!/bin/bash

zig build test
echo "Tests built"
cd tests
uv run pytest .
