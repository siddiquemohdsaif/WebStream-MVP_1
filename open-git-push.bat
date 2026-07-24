@echo off
set "current_path=%~dp0"
set "git_bash_path=%current_path:\=/%"
start "" "C:\Program Files\Git\git-bash.exe" -c "cd '%git_bash_path%' && bash --init-file ./git_push.sh"
