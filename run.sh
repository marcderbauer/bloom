# Doesn't need to query data, just use data that's already downloaded

echo"Installing requirements"
pip install -r requirements.txt
echo"Finished installing requirements"
sleep(3)

echo"Starting training. This may take a while..."
python3 main.py
echo"Finished training"
sleep(3)

echo"Trying to generate inference"
python3 inference.py Stupid
