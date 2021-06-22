curl "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured.zip" -o Structured.zip
unzip Structured.zip
rm Structured.zip
python scripts/data_preparation.py
mv Structured datasets/Structured-raw
python scripts/download_we.py