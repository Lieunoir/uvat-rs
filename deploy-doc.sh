#!/bin/bash

# Build the book
cargo doc

# If the gh-pages branch already exists, this will overwrite it
# so that the history is not kept, which can be very expensive.
git worktree add --orphan -B gh-pages gh-pages
cd target/doc/
rm -r ../../gh-pages
cp -r . ../../gh-pages
cd ../../
git config user.name "Deploy from CI"
git config user.email ""
cd gh-pages
git add -A
git commit -m 'Deploy doc'
git push origin +gh-pages
cd ..
