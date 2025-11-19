echo "A script to create , activate and install requirements.txt"
echo "..........."

virtualenv venv 

echo "creation of virtuealenv done....."
echo "Activation my env"
source venv/bin/activate

echo "..........."
echo "install requirements.txt"
pip install -r requirements.txt

sleep(2)
echo "install done"
echo "Creation Activation and install of library done"