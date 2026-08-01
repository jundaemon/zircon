#!/bin/bash

rm -rf tests/cases/
mkdir tests/cases/
echo "previous tests cleared"

zig build test
echo "new tests built"

cd tests
uv run pytest .
