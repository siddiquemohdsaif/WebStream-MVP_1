#!/bin/bash
eval "$(ssh-agent -s)"
ssh-add "/c/Users/shaqib pc/Documents/my_key_git"
git add .
git status
echo "Enter a commit message:"
read commit_message
git commit -m "$commit_message"
git push
