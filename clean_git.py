wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar -O bfg.jar
java -jar bfg.jar --delete-files rf_text_model.pkl
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push -u origin main --force