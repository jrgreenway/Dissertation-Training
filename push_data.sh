#!/bin/sh
git config --global user.email "james.r.greenway@live.co.uk"
git config --global user.name "James Greenway"

git add .
git commit -m "Add generated data"
git push https://$GIT_USERNAME:$GIT_TOKEN@github.com/your-username/your-repo.git Training